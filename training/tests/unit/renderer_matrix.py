"""Single source of truth for which renderers the QA harness covers.

Each ``RendererCase`` row binds a registered renderer name to the upstream
tokenizer it must match and to the capability flags that tell the harness
which invariants to run and which to skip. Adding a renderer to CI coverage
is a one-line change: append a row here.

Why the capability flags instead of per-renderer test files:

* ``supports_thinking`` / ``supports_tools`` gate the thinking- and
  tool-shaped scenarios so a renderer is never asked to round-trip a
  feature it does not implement.
* ``has_extension_property`` mirrors the renderer's own
  ``has_extension_property`` and gates the sequence-extension invariant
  (KV-cache-safe prefix growth across turns).
* ``supervised_hf_parity`` is False for renderers whose supervised
  rendering diverges from ``apply_chat_template(add_generation_prompt=
  False)`` *by design*: thinking-mode renderers that inject an empty
  ``<think></think>`` block or use a different assistant header for
  supervised vs generation (``<think>`` vs ``</think>``), renderers that
  append a synthetic terminal stop sentinel to supervised examples (GLM),
  or a custom non-Jinja encoder with no chat template (DeepSeek V4).
* ``observation_equals_generation`` is tracked *separately* because it is
  empirically independent of ``supervised_hf_parity``: kimi_k25 matches HF
  supervised bytes yet has ``ob != gen`` (it strips history), while qwen3
  has ``ob == gen`` yet diverges from the HF supervised render (which
  injects an empty ``<think></think>`` block the renderer omits). Only the
  ``ob == gen`` renderers exercise that leg of the consistency invariant.
* ``xfail_hf`` records a known, tracked HF byte-parity divergence so a fix
  flips the test green automatically.

Tokenizer ids use the public HF repo (all listed renderers, including
Gemma 4, resolve to ungated public repos). ``GEMMA4_MODEL_PATH`` may still
override Gemma 4 with a local checkpoint for offline dev boxes. When a
tokenizer cannot be loaded at all (network outage, preview-only repo), the
harness *skips* that case rather than erroring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RendererCase:
    """One (renderer, tokenizer) pair plus its harness capability flags.

    Attributes:
        renderer: Registered renderer name (cookbook-local or upstream).
        tokenizer_model: HF repo id / local path for the canonical tokenizer.
        hf_kwargs: Extra kwargs forwarded to ``apply_chat_template`` so the
            HF reference matches the renderer's mode (e.g. ``enable_thinking``,
            ``reasoning_effort``, ``preserve_thinking``).
        supports_thinking: Renderer understands chain-of-thought content.
        supports_tools: Renderer serializes tool calls / tool results.
        has_extension_property: Renderer satisfies the sequence-extension
            prefix property across assistant turns.
        has_hf_chat_template: The tokenizer exposes an HF chat template, so
            byte-parity invariants are meaningful.
        supervised_hf_parity: Supervised tokens are byte-identical to
            ``apply_chat_template(add_generation_prompt=False)`` (modulo the
            trailing-newline convention). Gates invariant 2.
        observation_equals_generation: The supervised observation (the
            weight-0 prefix of the last assistant turn) equals the generation
            prompt of the conversation prefix. Gates the ``ob == gen`` leg of
            invariant 3. This is deliberately *separate* from
            ``supervised_hf_parity``: the two are independent in practice
            (e.g. kimi_k25 matches HF bytes but has ob != gen because it
            strips history; qwen3 has ob == gen but injects an empty
            ``<think></think>`` block only in the HF supervised render).
        xfail_hf: When set, HF byte-parity is a known divergence; the
            generation-parity invariant xfails with this reason.
        tool_call_id_style: Native tool-call id convention for the
            render->parse round-trip (invariant 3) legs. ``None`` (default)
            leaves the scenario's OpenAI-style ids untouched. ``"kimi"``
            rewrites each tool-call id to ``functions.{name}:{index}`` because
            Kimi's wire format encodes the function NAME *inside* the id and
            carries no separate name field — feeding it an opaque id renders
            tokens from which the name is literally absent (unrecoverable by
            any parser), so the round-trip must use the id shape Kimi's own
            generation emits. Only affects invariant 3; the HF-parity legs
            (which render both sides from the same messages) are untouched.
    """

    renderer: str
    tokenizer_model: str
    hf_kwargs: dict[str, Any] = field(default_factory=dict)
    supports_thinking: bool = False
    supports_tools: bool = False
    has_extension_property: bool = False
    has_hf_chat_template: bool = True
    supervised_hf_parity: bool = False
    observation_equals_generation: bool = False
    xfail_hf: str | None = None
    tool_call_id_style: str | None = None


# Gemma 4's official instruct repos (``google/gemma-4-*-it``) are PUBLIC and
# ungated on the HF Hub (``AutoTokenizer.from_pretrained`` loads them
# unauthenticated), so gemma4 downloads in CI like every other public renderer.
# Default to the 31B repo because the renderer matches the gemma-4-31b-it /
# 26b-a4b-it template family; it DIVERGES from google/gemma-4-E2B-it (tool
# defs, tool responses, non-thinking thought suffix), so E2B would need its own
# xfail row rather than being the default reference.
#
# ``GEMMA4_MODEL_PATH`` still overrides with a local checkpoint dir (offline
# dev boxes; the dedicated ``test_gemma4_renderer.py`` uses the same env).
_GEMMA4_TOKENIZER = os.environ.get("GEMMA4_MODEL_PATH", "google/gemma-4-31B-it")


RENDERER_MATRIX: list[RendererCase] = [
    # -- GLM 5.x -----------------------------------------------------------
    # GLM strips historical <think> unconditionally (no extension property)
    # and appends a synthetic terminal role sentinel to supervised examples,
    # so supervised output is not byte-identical to apply_chat_template.
    RendererCase(
        renderer="glm5",
        tokenizer_model="zai-org/GLM-5.1",
        supports_thinking=True,
        supports_tools=True,
        has_extension_property=False,
        supervised_hf_parity=False,
        observation_equals_generation=False,
    ),
    RendererCase(
        renderer="glm_moe_dsa",
        tokenizer_model="zai-org/GLM-5.2",
        hf_kwargs={"reasoning_effort": "max"},
        supports_thinking=True,
        supports_tools=True,
        has_extension_property=False,
        supervised_hf_parity=False,
        observation_equals_generation=False,
    ),
    # -- Qwen3 -------------------------------------------------------------
    # Thinking-mode qwen3: the HF template injects an empty
    # `<think>\n\n</think>\n\n` block ahead of assistant content in the
    # supervised (add_generation_prompt=False) render, while the renderer
    # emits content directly — the same "different supervised header" class
    # as the qwen3_5/3_6 family (empirically confirmed against Qwen3-8B), so
    # supervised byte-parity and the ob==gen leg are disabled. Default
    # strips thinking from history -> no extension property.
    RendererCase(
        renderer="qwen3",
        tokenizer_model="Qwen/Qwen3-8B",
        supports_thinking=True,
        supports_tools=True,
        has_extension_property=False,
        supervised_hf_parity=False,
        observation_equals_generation=True,
    ),
    RendererCase(
        renderer="qwen3_disable_thinking",
        tokenizer_model="Qwen/Qwen3-8B",
        hf_kwargs={"enable_thinking": False},
        supports_thinking=False,
        supports_tools=True,
        has_extension_property=False,
        supervised_hf_parity=True,
        observation_equals_generation=True,
    ),
    # -- Qwen3.6 (aliases the qwen3_5 family) ------------------------------
    # The 3.5/3.6 thinking template uses </think> in supervised assistant
    # headers but <think> in the generation prompt, so observation !=
    # generation prompt by design -> supervised_hf_parity=False. Only the
    # preserve-thinking variant keeps history reasoning intact and thus
    # satisfies the extension property.
    RendererCase(
        renderer="qwen3_6",
        tokenizer_model="Qwen/Qwen3.6-27B",
        supports_thinking=True,
        supports_tools=True,
        has_extension_property=False,
        supervised_hf_parity=False,
        observation_equals_generation=False,
    ),
    RendererCase(
        renderer="qwen3_6_disable_thinking",
        tokenizer_model="Qwen/Qwen3.6-27B",
        hf_kwargs={"enable_thinking": False},
        supports_thinking=False,
        supports_tools=True,
        has_extension_property=False,
        supervised_hf_parity=False,
        observation_equals_generation=False,
    ),
    RendererCase(
        renderer="qwen3_6_preserve_thinking",
        tokenizer_model="Qwen/Qwen3.6-27B",
        hf_kwargs={"preserve_thinking": True},
        supports_thinking=True,
        supports_tools=True,
        has_extension_property=True,
        supervised_hf_parity=False,
        # Preserve mode emits an empty `<think>\n\n</think>\n\n` in the
        # supervised assistant header, while the generation prompt opens
        # `<think>`, so observation != generation prompt.
        observation_equals_generation=False,
    ),
    # -- Kimi K2.x ---------------------------------------------------------
    # kimi_k25 strips thinking from history (no extension) but keeps the
    # same supervised/generation header. kimi_k27_code preserves historical
    # thinking, but its current upstream base reports no extension property
    # and opens <think> only in generation prompts, so observation != generation.
    RendererCase(
        renderer="kimi_k25",
        tokenizer_model="moonshotai/Kimi-K2.5",
        supports_thinking=True,
        supports_tools=True,
        has_extension_property=False,
        supervised_hf_parity=True,
        observation_equals_generation=False,
        tool_call_id_style="kimi",
    ),
    RendererCase(
        renderer="kimi_k27_code",
        tokenizer_model="moonshotai/Kimi-K2.7-Code",
        supports_thinking=True,
        supports_tools=True,
        has_extension_property=False,
        supervised_hf_parity=True,
        observation_equals_generation=False,
        tool_call_id_style="kimi",
    ),
    # -- MiniMax M2 --------------------------------------------------------
    # Single-turn generation MATCHES the official HF MiniMax-M2 chat template
    # byte-for-byte (verified against MiniMaxAI/MiniMax-M2), so there is no
    # renderer-vs-template divergence to xfail here. The previously tracked
    # extra '\n' after <think> is a renderer-vs-DEPLOYED-MODEL divergence
    # (live probe against accounts/fireworks/models/minimax-m2p7), not a
    # renderer-vs-template one, and CPU HF parity cannot observe it; it is
    # tracked out-of-band (renderer-verifier-findings.md / live probe).
    # Any structural scenarios that DO diverge from the template (e.g.
    # leading-newline, consecutive-system, developer role) are mapped
    # per-(renderer, scenario) in the divergence maps
    # (:mod:`renderer_expected_divergences`) instead.
    RendererCase(
        renderer="minimax_m2",
        tokenizer_model="MiniMaxAI/MiniMax-M2",
        supports_thinking=True,
        supports_tools=True,
        has_extension_property=False,
        supervised_hf_parity=False,
        observation_equals_generation=False,
    ),
    # -- MiniMax M3 --------------------------------------------------------
    # Adaptive mode has no sampling-only assistant prefill: the model predicts
    # whether to open or close the thinking channel itself. Supervised and
    # generation prefixes therefore share the same bytes, and M3 preserves
    # complete assistant history across turns.
    RendererCase(
        renderer="minimax_m3",
        tokenizer_model="MiniMaxAI/MiniMax-M3",
        hf_kwargs={"thinking_mode": "adaptive"},
        supports_thinking=True,
        supports_tools=True,
        has_extension_property=True,
        supervised_hf_parity=True,
        observation_equals_generation=True,
    ),
    # -- Nemotron ----------------------------------------------------------
    # Thinking mode (always prepends <think></think>), so HF needs
    # enable_thinking=True. Supervised drops the trailing '\n' the HF
    # template emits after the final <|im_end|> -> supervised_hf_parity
    # is False. Default strips thinking from history -> no extension.
    RendererCase(
        renderer="nemotron3",
        tokenizer_model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        hf_kwargs={"enable_thinking": True},
        supports_thinking=True,
        supports_tools=True,
        has_extension_property=False,
        supervised_hf_parity=False,
        observation_equals_generation=False,
    ),
    # -- DeepSeek V4 -------------------------------------------------------
    # Custom non-Jinja encoder (no chat_template), so the HF-parity
    # invariants skip cleanly and only the consistency/extension invariants
    # exercise it. Thinking mode with history stripping -> no extension and
    # observation != generation prompt.
    RendererCase(
        renderer="deepseek_v4",
        tokenizer_model="deepseek-ai/DeepSeek-V4-Flash",
        supports_thinking=True,
        supports_tools=True,
        has_extension_property=False,
        has_hf_chat_template=False,
        supervised_hf_parity=False,
        observation_equals_generation=False,
    ),
    # -- Mistral / Ministral ----------------------------------------------
    # NOTE: main's ``training/renderer/__init__.py`` does not import
    # ``mistral.py``, so the "mistral" renderer is not registered on main and
    # cannot be exercised here. Add a row once it is wired into the package.
    # -- Gemma 4 -----------------------------------------------------------
    # Text-only, no thinking by default. The public 31B tokenizer is the
    # canonical reference; GEMMA4_MODEL_PATH is only an offline override.
    # observation_equals_generation is False: the non-thinking generation
    # prompt appends a sampling-only empty `<|channel>thought\n<channel|>`
    # marker (4 tokens) that the model fills and history then strips, so the
    # supervised observation is a strict prefix of the generation prompt, not
    # equal to it. Verified empirically against gemma-4-31b-it / 26b-a4b-it.
    # NOTE: no enable_thinking=True row here — the "gemma4" name registers
    # enable_thinking=False on this base (gemma4_thinking is not registered in
    # this harness); the gemma4 thinking-channel render/parse path is covered by
    # the separate gemma4 renderer-fix PR.
    RendererCase(
        renderer="gemma4",
        tokenizer_model=_GEMMA4_TOKENIZER,
        hf_kwargs={"enable_thinking": False},
        supports_thinking=False,
        supports_tools=True,
        # On main, "gemma4" resolves to Gemma4SplitRenderer (disaggregate
        # multi-turn SFT), which strips history and reports no extension
        # property, so the sequence-extension invariant does not apply.
        has_extension_property=False,
        supervised_hf_parity=True,
        observation_equals_generation=False,
    ),
    # Thinking-mode Gemma 4 (``gemma4_thinking`` registers enable_thinking=True).
    # Matches apply_chat_template(enable_thinking=True); the thinking channel is
    # emitted on the tool-calling turn and stripped from history, so history
    # reasoning is not preserved (no extension) and the supervised header
    # diverges from the generation prompt. Not in REQUIRED_RENDERERS because it
    # shares gemma4's tokenizer coverage; kept as a dedicated thinking row.
    RendererCase(
        renderer="gemma4_thinking",
        tokenizer_model=_GEMMA4_TOKENIZER,
        hf_kwargs={"enable_thinking": True},
        supports_thinking=True,
        supports_tools=True,
        has_extension_property=False,
        supervised_hf_parity=False,
        observation_equals_generation=False,
    ),
]


# Renderers whose tokenizer is public + ungated on the HF Hub and therefore
# MUST load whenever the harness runs with network access. The strict-mode
# coverage guard (see ``test_renderer_properties`` /
# ``test_required_renderers_load_in_strict_mode``) fails loudly if any of these
# silently drops to a skip — turning "a public model stopped being tested"
# (network regression, renamed repo, tokenizer-class breakage) into a red CI
# instead of an invisible green.
#
# Deliberately EXCLUDED (known to require an out-of-band asset, tracked, not
# silently ignored):
#   * kimi_k27_code — gated/preview tokenizer, not reliably public yet.
#   * deepseek_v4   — preview tokenizer ships no chat_template to diff against.
REQUIRED_RENDERERS: frozenset[str] = frozenset(
    {
        "glm5",
        "glm_moe_dsa",
        "qwen3",
        "qwen3_disable_thinking",
        "qwen3_6",
        "qwen3_6_disable_thinking",
        "qwen3_6_preserve_thinking",
        "kimi_k25",
        "minimax_m2",
        "minimax_m3",
        "nemotron3",
        "gemma4",
    }
)
