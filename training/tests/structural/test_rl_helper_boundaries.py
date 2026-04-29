"""Structural boundary tests for renderer-backed RL helpers and examples.

Enforces the architectural invariants from AC-6 / AC-7 / AC-8 / AC-11 /
AC-12:

* No new ``eval_protocol`` imports outside the
  ``examples/rl/ep_remote_grader/`` tree (snapshot-based: the set of
  importing files is locked, additions fail).
* No ``apply_chat_template(`` calls in new RL helper modules under
  ``utils/rl/`` or in any new renderer-backed example tree
  (``single_turn_*``, ``multi_turn_minimal_renderer``, ``multi_turn_tool``,
  ``remote_rollout``).
* No specific renderer-name references in trainer code
  (``utils/rl/{train,async_train,common,losses,metrics}.py`` plus other
  helper modules under ``utils/rl/``).
* No ``MessageEnv`` / ``ToolEnv`` Protocol definitions, ``ParseFailurePolicy``
  / ``TruncationPolicy`` enum definitions, ``parse_failure_policy`` /
  ``truncation_policy`` parameters, ``_apply_parse_policy`` helper,
  ``ExtensionPropertyUnavailable`` typed-error class, or
  ``cookbook.rl.parse_failure_total`` metric key under ``utils/rl/``.
* No cookbook-local ``sample_with_prompt_tokens`` shim under ``utils/rl/``
  (the SDK extension is the only sampler-extension surface).

These tests use AST parsing where possible and substring grepping where the
constraint is "no occurrence anywhere in the source" (matching the plan's
grep-form positive tests verbatim).  Tests are hermetic — they read source
files; they do not import anything that could pull in network state.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_UTILS_RL = _REPO_ROOT / "training" / "utils" / "rl"
_EXAMPLES_RL = _REPO_ROOT / "training" / "examples" / "rl"


def _python_files(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            out.append(root)
        elif root.is_dir():
            for p in root.rglob("*.py"):
                if "__pycache__" in p.parts:
                    continue
                out.append(p)
    return out


def _imports_module(path: Path, module_name: str) -> bool:
    """Return True iff ``path`` contains ``import <module>`` or
    ``from <module> ...`` as an actual import statement."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == module_name or alias.name.startswith(module_name + "."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == module_name or mod.startswith(module_name + "."):
                return True
    return False


# Snapshot of the eval_protocol-importing file set as of this change set.
# New files joining this list will fail the structural test until added
# here intentionally.  Per AC-6, existing eval_protocol imports in
# frozen_lake/, multihop_qa/, and tests/unit/ are out of scope; only
# ep_remote_grader/ is "the only allowed example tree" for NEW EP
# integration code in this change set.
_EVAL_PROTOCOL_IMPORTERS = frozenset({
    "training/examples/rl/ep_remote_grader/ep_service.py",
    "training/examples/rl/ep_remote_grader/grader.py",
    "training/examples/rl/frozen_lake/train_frozen_lake.py",
    # FrozenLakeRolloutService now lives next to FrozenLakeToolRolloutProcessor
    # in frozen_lake_rollout.py — no NEW eval_protocol-importing file is added
    # by the renderer-reuse change set.
    "training/examples/rl/frozen_lake/frozen_lake_rollout.py",
    "training/examples/rl/frozen_lake/verify_rollout.py",
    "training/examples/rl/frozen_lake/frozen_lake_schema.py",
    "training/examples/multihop_qa/train_multihop_qa_igpo.py",
    # MultiHopQARolloutService now lives next to MultiHopQARolloutProcessor
    # in multihop_qa_rollout.py — no NEW eval_protocol-importing file is added.
    "training/examples/multihop_qa/multihop_qa_rollout.py",
    # Pre-existing test files importing EP (out of scope per AC-6 wording).
    "training/tests/unit/test_frozen_lake_verify_validation.py",
    "training/tests/unit/test_frozen_lake_visual_rollout.py",
    "training/tests/unit/test_train_frozen_lake.py",
})


# ---------------------------------------------------------------------------
# AC-6: no NEW eval_protocol imports outside ep_remote_grader/ for this change set
# ---------------------------------------------------------------------------


class TestEvalProtocolBoundary:
    def test_no_new_eval_protocol_importers(self):
        actual: set[str] = set()
        for f in _python_files([_REPO_ROOT / "training"]):
            if _imports_module(f, "eval_protocol"):
                rel = str(f.relative_to(_REPO_ROOT))
                actual.add(rel)

        new = actual - _EVAL_PROTOCOL_IMPORTERS
        gone = _EVAL_PROTOCOL_IMPORTERS - actual
        assert not new, (
            f"NEW eval_protocol imports detected (not allowed by AC-6): {sorted(new)}.\n"
            "If this is intentional, add the path(s) to "
            "_EVAL_PROTOCOL_IMPORTERS and document the reason."
        )
        # Disappeared files (rename / removal) are fine — this test is about
        # net-new additions only.  Touch the value so flake8 doesn't flag it.
        _ = gone


# ---------------------------------------------------------------------------
# AC-8 / AC-9 / AC-11: no apply_chat_template in new helper modules / examples
# ---------------------------------------------------------------------------


_NEW_RENDERER_BACKED_TREES = [
    _UTILS_RL / "renderer_rollout.py",
    _EXAMPLES_RL / "single_turn_sync_on_policy",
    _EXAMPLES_RL / "single_turn_async",
    _EXAMPLES_RL / "multi_turn_minimal_renderer",
    _EXAMPLES_RL / "multi_turn_tool",
    _EXAMPLES_RL / "remote_rollout",
]


class TestNoChatTemplateRenderingClientSide:
    def test_no_apply_chat_template_in_new_helper_modules(self):
        offenders: list[str] = []
        for f in _python_files(_NEW_RENDERER_BACKED_TREES):
            if "test_" in f.name:
                # Test files may mention the string in assertion error
                # messages; this is fine.  We constrain implementation files.
                continue
            text = f.read_text()
            if "apply_chat_template(" in text:
                offenders.append(str(f.relative_to(_REPO_ROOT)))
        assert not offenders, (
            f"apply_chat_template( found in renderer-backed module(s): {offenders}.  "
            "Renderer-backed rollouts must consume the renderer's "
            "build_generation_prompt + the SDK's sample_with_prompt_tokens; "
            "client-side chat-template rendering re-introduces tokenizer drift."
        )

    def test_no_sample_with_tokens_messages_kwarg_in_new_code(self):
        offenders: list[str] = []
        for f in _python_files(_NEW_RENDERER_BACKED_TREES):
            if "test_" in f.name:
                continue
            text = f.read_text()
            if "sample_with_tokens(messages=" in text:
                offenders.append(str(f.relative_to(_REPO_ROOT)))
        assert not offenders, (
            f"sample_with_tokens(messages=...) found in new code: {offenders}.  "
            "New renderer-backed code must use sample_with_prompt_tokens; "
            "the legacy method is preserved for legacy callers only."
        )


# ---------------------------------------------------------------------------
# AC-12: no specific renderer names in trainer modules + new helper modules
# ---------------------------------------------------------------------------


_TRAINER_MODULES = [
    _UTILS_RL / "train.py",
    _UTILS_RL / "async_train.py",
    _UTILS_RL / "common.py",
    _UTILS_RL / "losses.py",
    _UTILS_RL / "metrics.py",
]

# Modules whose declared purpose is renderer dispatch (out of this rule's
# scope).  Empty for now.
_RENDERER_DISPATCH_MODULES: set[Path] = set()

_RENDERER_NAME_PATTERN = re.compile(
    r"\b(gemma4|qwen3|glm5|minimax_m2|nemotron|kimi_k2|kimi_k25|kimi_k26|deepseek_v3|llama3)\b"
)


class TestNoRendererNamesInTrainerOrUtilsRL:
    def test_trainer_modules_renderer_name_agnostic(self):
        offenders: list[tuple[str, str]] = []
        for f in _TRAINER_MODULES:
            if not f.exists():
                continue
            text = f.read_text()
            for m in _RENDERER_NAME_PATTERN.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                offenders.append((str(f.relative_to(_REPO_ROOT)), f"L{line_no}: {m.group(0)}"))
        assert not offenders, (
            f"Renderer-name reference(s) in trainer code: {offenders}.  "
            "Trainer code must stay renderer-name-agnostic per AC-12."
        )

    def test_new_utils_rl_modules_renderer_name_agnostic(self):
        offenders: list[tuple[str, str]] = []
        for f in _python_files([_UTILS_RL]):
            if f in _RENDERER_DISPATCH_MODULES:
                continue
            text = f.read_text()
            for m in _RENDERER_NAME_PATTERN.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                offenders.append((str(f.relative_to(_REPO_ROOT)), f"L{line_no}: {m.group(0)}"))
        # Pre-existing helper modules in utils/rl/ may reference renderer
        # names today; this test scopes the constraint to renderer_rollout.py
        # (the new module landed by this change set).  Existing modules are
        # snapshot-allowed; new modules are checked.
        new_offenders = [(p, m) for (p, m) in offenders if "renderer_rollout" in p]
        assert not new_offenders, (
            f"Renderer-name reference(s) in new utils/rl helper modules: {new_offenders}"
        )


# ---------------------------------------------------------------------------
# AC-7 / AC-12: no exported Protocols / enums / typed errors / metric keys
# ---------------------------------------------------------------------------


_FORBIDDEN_SYMBOL_PATTERN = re.compile(
    r"\b(class\s+MessageEnv\b|class\s+ToolEnv\b|"
    r"class\s+ParseFailurePolicy\b|class\s+TruncationPolicy\b|"
    r"\bparse_failure_policy\b|\btruncation_policy\b|"
    r"\bExtensionPropertyUnavailable\b|"
    r"\b_apply_parse_policy\b|"
    r"cookbook\.rl\.parse_failure_total)"
)


class TestNoForbiddenExports:
    def test_utils_rl_exports_no_protocols_enums_typed_errors_or_metrics(self):
        offenders: list[tuple[str, str]] = []
        for f in _python_files([_UTILS_RL]):
            text = f.read_text()
            for m in _FORBIDDEN_SYMBOL_PATTERN.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                offenders.append((str(f.relative_to(_REPO_ROOT)), f"L{line_no}: {m.group(0)}"))
        assert not offenders, (
            f"Forbidden symbol(s) in utils/rl/: {offenders}.  "
            "Multi-turn / tool flows must live as concrete examples; "
            "parse-failure / truncation handling is user code."
        )


# ---------------------------------------------------------------------------
# AC-12: no cookbook-local sample_with_prompt_tokens shim under utils/rl/
# ---------------------------------------------------------------------------


class TestSharedRendererTurnLoop:
    """AC-4 sub-rule: the multi-turn-minimal-renderer and multi-turn-tool
    examples share exactly ONE renderer/sampler/assembly inner loop."""

    _MULTI_TURN_FILES = [
        _EXAMPLES_RL / "multi_turn_minimal_renderer" / "rollout.py",
        _EXAMPLES_RL / "multi_turn_tool" / "rollout.py",
    ]

    def test_both_rollouts_import_the_shared_step(self):
        for f in self._MULTI_TURN_FILES:
            tree = ast.parse(f.read_text())
            imports_shared = False
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and node.module.endswith("_renderer_turn_loop"):
                        names = {alias.name for alias in node.names}
                        if "renderer_turn_step" in names:
                            imports_shared = True
                            break
            assert imports_shared, (
                f"{f.relative_to(_REPO_ROOT)} must import renderer_turn_step "
                "from training.examples.rl._renderer_turn_loop (shared per-turn loop)."
            )

    def test_neither_rollout_calls_assembler_add_call_directly(self):
        # The shared step is the only place that calls TrajectoryAssembler.add_call
        # for the renderer-backed multi-turn flatten loop.  Direct calls in the
        # example files would re-introduce the duplicated inner loop.
        for f in self._MULTI_TURN_FILES:
            tree = ast.parse(f.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr == "add_call":
                        raise AssertionError(
                            f"{f.relative_to(_REPO_ROOT)} calls "
                            "assembler.add_call(...) directly; this should be in "
                            "the shared _renderer_turn_loop step."
                        )


class TestPerTurnVersionPlumbing:
    """AC-13 follow-through: the shared ``renderer_turn_step`` must thread
    per-turn versions into ``InferenceCall.output_versions`` so assistant
    spans carry their call-time deployment version through to the trainer."""

    def test_renderer_turn_step_threads_output_versions(self):
        path = _EXAMPLES_RL / "_renderer_turn_loop.py"
        text = path.read_text()
        assert "output_versions=" in text, (
            "_renderer_turn_loop.py must pass output_versions into InferenceCall(...) "
            "so per-turn versions are preserved in the assembled trajectory."
        )

    def test_multi_turn_examples_use_pack_assembled_to_sample(self):
        # The assembled-aware packer is what preserves per-turn versions on
        # the emitted RolloutSample.  Both multi-turn examples must call it
        # instead of pack_payload_to_sample(version=<single scalar>).
        for f in (
            _EXAMPLES_RL / "multi_turn_minimal_renderer" / "rollout.py",
            _EXAMPLES_RL / "multi_turn_tool" / "rollout.py",
        ):
            tree = ast.parse(f.read_text())
            calls_assembled = False
            calls_payload = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    name = None
                    if isinstance(node.func, ast.Name):
                        name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        name = node.func.attr
                    if name == "pack_assembled_to_sample":
                        calls_assembled = True
                    elif name == "pack_payload_to_sample":
                        calls_payload = True
            assert calls_assembled, (
                f"{f.relative_to(_REPO_ROOT)} must call pack_assembled_to_sample "
                "from _renderer_turn_loop."
            )
            assert not calls_payload, (
                f"{f.relative_to(_REPO_ROOT)} should not call pack_payload_to_sample "
                "(it collapses per-turn versions onto a single terminal scalar)."
            )


class TestMigratedLegacyExamples:
    """AC-11 follow-through: gsm8k_async, deepmath, frozen_lake, multihop_qa
    each expose the cookbook-native rollout surface.  gsm8k_async +
    deepmath use ``single_turn_renderer_rollout`` directly; frozen_lake +
    multihop_qa expose a ``RolloutService`` adapter that the trainer can
    consume via ``make_remote_rollout_fn(...)``."""

    def test_gsm8k_async_uses_renderer_rollout(self):
        text = (_EXAMPLES_RL / "gsm8k_async" / "train.py").read_text()
        assert "single_turn_renderer_rollout" in text
        assert "sample_with_tokens(messages=" not in text

    def test_deepmath_uses_renderer_rollout_and_pluggable_rollout_fn(self):
        text = (_EXAMPLES_RL / "deepmath" / "train_deepmath.py").read_text()
        assert "single_turn_renderer_rollout" in text
        # The legacy mutation pattern must be gone.
        assert not re.search(r"\brl_loop\.reward_fn\s*=", text)

    def test_frozen_lake_exposes_rollout_service_adapter(self):
        # The adapter lives next to the processor (already EP-aware) so no
        # new eval_protocol-importing file is added.
        path = _EXAMPLES_RL / "frozen_lake" / "frozen_lake_rollout.py"
        assert "class FrozenLakeRolloutService" in path.read_text()
        # No standalone rollout_service.py file should exist (AC-6 boundary).
        assert not (_EXAMPLES_RL / "frozen_lake" / "rollout_service.py").exists()

    def test_multihop_qa_exposes_rollout_service_adapter(self):
        path = _REPO_ROOT / "training" / "examples" / "multihop_qa" / "multihop_qa_rollout.py"
        assert "class MultiHopQARolloutService" in path.read_text()
        assert not (_REPO_ROOT / "training" / "examples" / "multihop_qa" / "rollout_service.py").exists()

    def test_frozen_lake_train_does_not_construct_evaluation_row(self):
        """The trainer-facing sampling path must consume the cookbook
        ``RolloutService`` surface.  Direct ``EvaluationRow(...)`` construction
        or processor invocation inside ``sample_one_prompt`` would re-introduce
        the legacy EP-driven loop Codex flagged in Round 2."""
        path = _EXAMPLES_RL / "frozen_lake" / "train_frozen_lake.py"
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "sample_one_prompt":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        name = None
                        if isinstance(sub.func, ast.Name):
                            name = sub.func.id
                        elif isinstance(sub.func, ast.Attribute):
                            name = sub.func.attr
                        if name in {"EvaluationRow", "FrozenLakeToolRolloutProcessor"}:
                            raise AssertionError(
                                f"frozen_lake/train_frozen_lake.py::sample_one_prompt "
                                f"calls {name}(...) directly; sampling must route "
                                "through make_remote_rollout_fn / "
                                "FrozenLakeRolloutService instead."
                            )

    def test_multihop_qa_train_does_not_construct_evaluation_row(self):
        path = _REPO_ROOT / "training" / "examples" / "multihop_qa" / "train_multihop_qa_igpo.py"
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "sample_one_prompt":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call):
                        name = None
                        if isinstance(sub.func, ast.Name):
                            name = sub.func.id
                        elif isinstance(sub.func, ast.Attribute):
                            name = sub.func.attr
                        if name in {"EvaluationRow", "MultiHopQARolloutProcessor"}:
                            raise AssertionError(
                                f"multihop_qa/train_multihop_qa_igpo.py::sample_one_prompt "
                                f"calls {name}(...) directly; sampling must route "
                                "through MultiHopQARolloutService instead."
                            )


class TestNoSyntheticPromptTokenSynthesisInEPRequestPath:
    def test_ep_service_does_not_synthesize_prompt_tokens(self):
        # AC-6: ep_remote_grader/ep_service.py must build request-side
        # prompt token IDs from the renderer, not from chr()/ord()-style
        # synthesis.  We grep for the canonical synthesis fingerprint and
        # for any ord()-driven token synthesis pattern in the request side.
        path = (_REPO_ROOT / "training" / "examples" / "rl" / "ep_remote_grader"
                / "ep_service.py")
        text = path.read_text()
        # The exact synthesis pattern documented in the original plan as
        # forbidden.
        assert "2000 + (ord(c) % 100)" not in text, (
            "ep_remote_grader/ep_service.py still contains the forbidden "
            "[2000 + (ord(c) % 100) ...] prompt-token synthesis path."
        )
        # Defense in depth: no `chr(...)` or `ord(c) % ...` token synthesis
        # remains in the request-side path.
        forbidden_re = re.compile(r"ord\(\s*[A-Za-z_]+\s*\)\s*%\s*\d+")
        assert not forbidden_re.search(text), (
            "ep_remote_grader/ep_service.py still contains an ord()-based "
            "token synthesis pattern; the request side must use the renderer."
        )


class TestNoCookbookLocalSamplerShim:
    def test_no_sample_with_prompt_tokens_definition_under_utils_rl(self):
        # The SDK extension in fireworks-ai-python is the only sampler-
        # extension surface.  utils/rl/ may *call* sample_with_prompt_tokens
        # via dependency injection but must not redefine it.
        offenders: list[tuple[str, int]] = []
        for f in _python_files([_UTILS_RL]):
            try:
                tree = ast.parse(f.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == "sample_with_prompt_tokens":
                        offenders.append(
                            (str(f.relative_to(_REPO_ROOT)), node.lineno)
                        )
        assert not offenders, (
            f"Cookbook-local sample_with_prompt_tokens definition in utils/rl/: {offenders}.  "
            "The SDK extension (DeploymentSampler.sample_with_prompt_tokens) "
            "is the only sampler-extension surface."
        )
