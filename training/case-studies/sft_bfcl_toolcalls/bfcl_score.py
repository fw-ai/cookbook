#!/usr/bin/env python3
"""Generate tool calls against a Fireworks model, then score them BFCL's way.

Two halves, deliberately separated:

`generate_category()` is ours -- a plain OpenAI-SDK loop against the Fireworks
endpoint. It stays in this repo, and stays readable, because "how do I get a
model to emit a tool call" is the thing a reader is here to learn.

`score_category()` is *not* ours. It calls `bfcl_eval`'s own `ast_checker`, the
same function the leaderboard runs, so the numbers this notebook prints are
comparable to the published board rather than to a scorer we invented. We also
dump results in BFCL's on-disk format so `bfcl evaluate` can be run as a
cross-check (see `write_bfcl_results`).

On the AST checker's `model_name` argument: it looks the key up in
`MODEL_CONFIG_MAPPING` to decide whether to rewrite `.` to `_` in the ground
truth, and raises `KeyError` on anything unregistered. Rather than construct and
inject a `ModelConfig` for our fine-tune -- which couples this file to a private
dataclass signature -- we borrow a registered key whose rewrite behaviour matches
ours. That is the only thing the key is used for here.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from bfcl_data import LANGUAGE, NON_LIVE_AST, to_openai_tools, user_prompt

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

# A registered FC key with underscore_to_dot=True, matching our own `.`->`_`
# rewrite. Used only to select ground-truth name normalization in ast_checker.
AST_NAME_STYLE_KEY = "gpt-4.1-2025-04-14-FC"


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def _call_one(
    client,
    model: str,
    row: dict,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    reasoning_effort: str | None = None,
) -> dict:
    """One row -> one BFCL result record.

    `result` follows BFCL's FC-mode shape: a list of `{func_name: json_arg_string}`.
    An empty list means "no call", which is exactly what the irrelevance
    categories are scored on.
    """
    tools = to_openai_tools(row.get("function", []))
    # Must match how the model was TRAINED to be prompted. A renderer that
    # prefills a closed `<think></think>` block trains the model to start its
    # answer after one; serve it without and it generates reasoning instead of
    # a tool call, or emits a fragment the tool-call parser rejects. See
    # `assert_thinking_parity` for the invariant this has to satisfy.
    #
    # `enable_thinking=False` is accepted everywhere (it maps to
    # `reasoning_effort="none"`), but on some models "none" still reasons at
    # full length -- so it is not a reliable way to keep a response short. Pass
    # an explicit `reasoning_effort` for those; see `reasoning_control_for`.
    if reasoning_effort is not None:
        extra_body = {"reasoning_effort": reasoning_effort}
    elif enable_thinking:
        extra_body = {}
    else:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    last_error = ""
    for attempt in range(9):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_prompt(row)}],
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body=extra_body,
                timeout=120,
            )
            msg = resp.choices[0].message
            calls = [
                {tc.function.name: tc.function.arguments}
                for tc in (msg.tool_calls or [])
            ]
            usage = resp.usage
            return {
                "id": row["id"],
                "result": calls,
                "input_token_count": getattr(usage, "prompt_tokens", 0),
                "output_token_count": getattr(usage, "completion_tokens", 0),
                # A truncated response yields no parseable call, which scores as
                # a deliberate abstention. Record the reason so
                # `truncated_rows()` can tell the two apart after the fact.
                "finish_reason": resp.choices[0].finish_reason,
                "latency": round(time.time() - t0, 3),
            }
        except Exception as exc:  # noqa: BLE001
            text = str(exc)
            last_error = text[:200]
            # On-demand deployments return 503 while a replica spins up. Retry
            # those; surface everything else. A row that never succeeds is scored
            # as "no call", which is a real (wrong) answer for AST categories.
            if ("503" in text or "scaling up" in text or "scaled to zero" in text) and attempt < 8:
                time.sleep(15)
                continue
            if attempt >= 8:
                break
            print(f"  {row['id']}: attempt {attempt + 1}/9 failed: {text[:120]}")
            time.sleep(2 * (attempt + 1))
    return {"id": row["id"], "result": [], "error": f"generation_failed: {last_error}"}


def generate_category(
    model: str,
    rows: list[dict],
    *,
    api_key: str | None = None,
    base_url: str = FIREWORKS_BASE_URL,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    concurrency: int = 8,
    enable_thinking: bool = False,
    reasoning_effort: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Run every row of one category through `model`. Order is preserved."""
    # lazy: `openai` is only needed to generate, not to score an existing run.
    from openai import OpenAI

    client = OpenAI(api_key=api_key or os.environ["FIREWORKS_API_KEY"], base_url=base_url)
    out: list[dict] = [None] * len(rows)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _call_one,
                client,
                model,
                row,
                max_tokens,
                temperature,
                enable_thinking,
                reasoning_effort,
            ): i
            for i, row in enumerate(rows)
        }
        for done, fut in enumerate(as_completed(futures), start=1):
            out[futures[fut]] = fut.result()
            if progress and done % 25 == 0:
                progress(done, len(rows))
    return out


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def _decode(result: list[dict]) -> tuple[list[dict], int]:
    """BFCL FC result -> the decoded `[{name: {arg: value}}]` the checker wants.

    A call whose argument string fails to parse is NOT counted as a decoded
    call -- the official harness treats a decode failure as "no parseable
    call" (`_evaluate_single_relevance_entry` sets contain_func_call=False on
    any decode_ast exception), so inventing `{name: {}}` here would score a
    malformed call as a real one and under-count irrelevance. The second
    return value is the number of malformed calls, so `score_category` can
    still fail an AST row that emitted an unparseable call.
    """
    decoded = []
    n_failed = 0
    for call in result:
        for name, args in call.items():
            try:
                decoded.append({name: json.loads(args) if isinstance(args, str) else args})
            except json.JSONDecodeError:
                n_failed += 1
    return decoded, n_failed


def _language_enum(category: str):
    """Our string language keys -> the `Language` enum `ast_checker` compares against.

    `Language.PYTHON` is a plain `Enum` whose value happens to be `"python"`, so
    `Language.PYTHON == "python"` is **False**. Passing the bare string makes
    `simple_function_checker` raise `ValueError: Unsupported language: python` on
    every row that gets far enough to be type-checked -- which silently scores
    those rows as wrong instead of erroring out loudly.
    """
    # lazy: part of the `bfcl-eval` dependency tree, only needed when scoring.
    from bfcl_eval.constants.enums import Language

    return {
        "python": Language.PYTHON,
        "java": Language.JAVA,
        "javascript": Language.JAVASCRIPT,
    }[LANGUAGE.get(category, "python")]


def score_category(category: str, rows: list[dict], results: list[dict], answers: dict) -> dict:
    """Accuracy for one category, using BFCL's own checker.

    AST categories go through `ast_checker`. Irrelevance is scored structurally,
    the way `eval_runner._evaluate_single_relevance_entry` does it: a row is
    correct when the model produced *no* parseable call -- a call with
    unparseable JSON arguments counts as no parseable call. Argument values are
    never inspected for irrelevance -- only whether the model abstained.
    """
    # lazy: `bfcl-eval` pulls a large dependency tree; only scoring needs it.
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

    language = _language_enum(category)

    by_id = {r["id"]: r for r in results}
    correct, failures = 0, []

    for row in rows:
        res = by_id.get(row["id"], {"result": []})
        decoded, n_malformed = _decode(res.get("result", []))

        if category not in NON_LIVE_AST:  # irrelevance
            # A malformed-argument call is not a parseable call (see _decode),
            # matching the official scorer's decode-failure semantics. The
            # decoder is all-or-nothing over the response, so a row mixing one
            # valid and one malformed call is a decode failure overall -- i.e.
            # a correct abstention here, not a call.
            ok = len(decoded) == 0 or n_malformed > 0
            detail = {"error_type": "irrelevance_error:decoder_success"}
        else:
            try:
                verdict = ast_checker(
                    row["function"],
                    decoded,
                    answers[row["id"]],
                    language,
                    category,
                    AST_NAME_STYLE_KEY,
                )
                ok = bool(verdict.get("valid")) and n_malformed == 0
                if n_malformed:
                    detail = {**verdict, "error_type": "relevance_error:decoder_failed_malformed_args"}
                else:
                    detail = verdict
            except Exception as exc:  # noqa: BLE001 - a checker crash is a failed row
                ok, detail = False, {"error_type": f"checker_exception:{type(exc).__name__}"}
        if ok:
            correct += 1
        elif len(failures) < 5:
            failures.append({"id": row["id"], "detail": detail})

    return {
        "category": category,
        "accuracy": correct / len(rows) if rows else 0.0,
        "correct_count": correct,
        "total_count": len(rows),
        "sample_failures": failures,
    }


def summarize(scores: dict[str, dict]) -> dict[str, float]:
    """Roll per-category accuracies up the way BFCL's summary CSVs do.

    `simple_ast` is the *unweighted* mean of the three languages, and non-live
    overall is the unweighted mean of {simple_ast, multiple, parallel,
    parallel_multiple}. Irrelevance is deliberately NOT folded into the AST
    number -- it is reported alongside, because a model can buy irrelevance by
    refusing to call anything.
    """
    def acc(cat: str) -> float:
        return scores[cat]["accuracy"] * 100 if cat in scores else 0.0

    simple_ast = sum(acc(c) for c in ("simple_python", "simple_java", "simple_javascript")) / 3
    non_live = (simple_ast + acc("multiple") + acc("parallel") + acc("parallel_multiple")) / 4
    return {
        "simple_ast": round(simple_ast, 2),
        "non_live_ast": round(non_live, 2),
        "irrelevance": round(acc("irrelevance"), 2),
    }


def generation_failures(results: list[dict]) -> int:
    """Rows where generation never succeeded -- empty by failure, not by choice.

    This has to be counted explicitly, and loudly, because of how a failure is
    represented. `_call_one` gives up after its retries and returns an empty
    `result`, which is byte-identical on disk to a model that deliberately emitted
    no call. For an AST category that scores as a wrong answer; for an irrelevance
    row it scores as *correct*. So a transient 503 silently becomes model behaviour.

    It is not hypothetical. One sweep here lost 32 of 100 `simple_java` rows this
    way and reported 22.00% against 36.00% on the two clean repeats -- a 14-point
    "regression" that was entirely infrastructure. An earlier 200-row sweep lost 8
    rows and produced a 3.00pp gap that we mistook for the harness's noise floor,
    and then reasoned from.

    Detected two ways so the check survives a round-trip through disk: the explicit
    `error` key, and the absence of the usage fields that only a successful call sets.
    """
    return sum(1 for r in results if r.get("error") or "output_token_count" not in r)


def summarize_repeats(summaries: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and observed range across repeated evaluations of the *same* model.

    One sweep is a point estimate. Reporting `mean [min, max]` over a few repeats is
    what separates "+3.69" from "+3.69, and here is how much of that the harness
    could have produced on its own". A delta smaller than the wider of the two
    ranges is not a result.

    Measured here, on clean sweeps: this harness is very reproducible. Six
    evaluations of one unchanged model -- two deployments, two sessions -- spanned
    [63.79, 64.42] on `non_live_ast`, and `parallel` and `irrelevance` came back
    *identical* every time. `temperature=0.0` really does hold. Which means the thing
    worth guarding against is not sampling nondeterminism -- it is a failed sweep
    scored as if it succeeded. See `generation_failures`.
    """
    if not summaries:
        raise ValueError("need at least one summary to aggregate")
    keys = summaries[0].keys()
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        vals = [s[key] for s in summaries]
        out[key] = {
            "mean": round(sum(vals) / len(vals), 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "spread": round(max(vals) - min(vals), 2),
            "n": len(vals),
        }
    return out


def write_bfcl_results(
    model_label: str, category: str, results: list[dict], project_root: str | Path
) -> Path:
    """Write results where `bfcl evaluate` expects them, for an optional cross-check.

        bfcl evaluate --model <model_label> --test-category non_live

    The `error` key is written out rather than stripped. It is not part of BFCL's
    format, but the evaluator only reads `id` and `result`, and keeping it is what
    lets `generation_failures` recognise a contaminated file after a reload instead
    of scoring it as real model output.
    """
    out_dir = Path(project_root) / "result" / model_label / "non_live"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"BFCL_v4_{category}_result.json"
    with path.open("w", encoding="utf-8") as fh:
        for rec in results:
            fh.write(json.dumps(rec) + "\n")
    return path


def call_count_report(
    project_root: str | Path, base_label: str, tuned_label: str, categories: list[str]
) -> list[dict]:
    """Compare *how many* calls each model emitted, per category.

    When an SFT run moves a score the wrong way, this is the first thing to look
    at, because it separates two failures that need opposite fixes:

    - **zero-call rate up on single-call categories** -- the model learned to
      refuse. Usually too many negatives, or negatives that all share one refusal
      string. Lower `N_NEGATIVE` or vary the wording.
    - **multi-call rate up on single-call categories** -- the model learned to
      over-call, because the training mix is heavy on parallel rows. BFCL's
      `simple_*` and `multiple` checkers fail immediately with `wrong_count` when
      the call count is not exactly 1, so this shows up as a large score drop
      with no argument errors at all.

    Read it against the expected count: 1 for `simple_*` and `multiple`, more
    than 1 for the `parallel*` pair, and 0 for `irrelevance`.
    """
    def _load(label: str, cat: str) -> dict:
        path = Path(project_root) / "result" / label / "non_live" / f"BFCL_v4_{cat}_result.json"
        return {
            json.loads(line)["id"]: json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    expected = {"irrelevance": "0", "parallel": ">1", "parallel_multiple": ">1"}
    rows = []
    for cat in categories:
        base, tuned = _load(base_label, cat), _load(tuned_label, cat)
        bn = [len(v["result"]) for v in base.values()]
        tn = [len(v["result"]) for v in tuned.values()]
        rows.append({
            "category": cat,
            "expected_calls": expected.get(cat, "1"),
            "base_zero_pct": round(100 * sum(1 for x in bn if x == 0) / len(bn), 1),
            "tuned_zero_pct": round(100 * sum(1 for x in tn if x == 0) / len(tn), 1),
            "base_multi_pct": round(100 * sum(1 for x in bn if x > 1) / len(bn), 1),
            "tuned_multi_pct": round(100 * sum(1 for x in tn if x > 1) / len(tn), 1),
        })
    return rows


def load_results(project_root: str | Path, label: str, category: str) -> list[dict] | None:
    """Read back a previously written result file, or None if it does not exist.

    Lets a re-run reuse an earlier base-model evaluation instead of paying to
    deploy and re-score the same rows. The base model does not change while you
    iterate on training data, so its scores do not either.
    """
    path = Path(project_root) / "result" / label / "non_live" / f"BFCL_v4_{category}_result.json"
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def thinking_enabled_for(renderer_name: str) -> bool:
    """Whether to let the served template open a thinking block, given the renderer.

    Training and inference have to agree on the assistant-turn prefix, and the
    renderer is what decides it:

        qwen3                   ->  <|im_start|>assistant\\n
        qwen3_disable_thinking  ->  <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n

    Train with the second and serve with the first and the model is dropped into
    a prompt it never saw. It reasons where it learned to emit a call, runs to the
    token cap, or produces a fragment like `{"name": ...}\\n</tool_call>` with no
    opening tag -- correct arguments, unparseable envelope, scored as no-call.

    Deriving the flag from the renderer name keeps the two from drifting apart,
    which is the whole point: as two independent knobs they silently disagree,
    and the symptom looks like a data problem rather than a prompt problem.
    """
    return "disable_thinking" not in renderer_name


# Renderers whose model family reliably stops reasoning on
# `chat_template_kwargs.enable_thinking=False`. For everything else, require an
# explicit `reasoning_effort` tier instead -- see `reasoning_control_for`.
_ENABLE_THINKING_RENDERERS = ("qwen3", "qwen2_5")


def reasoning_control_for(renderer_name: str) -> str:
    """Which knob to use to control reasoning for this renderer's model family.

    Returns ``"enable_thinking"`` or ``"reasoning_effort"``.

    Fireworks does map ``chat_template_kwargs={"enable_thinking": False}`` onto
    ``reasoning_effort="none"``, so the Qwen-style flag is not rejected. The
    problem is that ``"none"`` does not suppress reasoning on every model.
    Measured on `muse_glimmer` over four prompts, mean output tokens were:
    default 339, ``enable_thinking=False`` 359, ``"none"`` 337, ``"low"`` 170.
    Only an explicit low tier actually shortens the response.

    That decides whether a sweep measures the model or its token cap: a
    full-length reasoning trace exhausts ``max_tokens``, the truncated response
    carries no parseable call, and the row scores as a deliberate abstention --
    correct on irrelevance, wrong on an AST category, and invisible in the
    accuracy number either way. Check ``truncated_rows()``, not the config.
    """
    if renderer_name.startswith(_ENABLE_THINKING_RENDERERS):
        return "enable_thinking"
    return "reasoning_effort"


def assert_thinking_parity(
    renderer_name: str, enable_thinking: bool, reasoning_effort: str | None = None
) -> None:
    """Fail loudly when the eval prompt cannot match the training prompt.

    Checks two separate things:

    1. The thinking flag agrees with the renderer (the original invariant).
    2. This renderer's model family is one where that flag is actually enough to
       keep responses short. Without (2) the guard passes, the model reasons to
       its token cap anyway, and a truncation artifact is recorded as model
       behaviour.

    Passing this is necessary but **not** sufficient: it validates configuration,
    not what the server did. Use ``truncated_rows()`` on the results for that.
    """
    control = reasoning_control_for(renderer_name)
    if control == "reasoning_effort":
        if reasoning_effort is None:
            raise ValueError(
                f"renderer {renderer_name!r} needs an explicit `reasoning_effort`. "
                "`enable_thinking=False` maps to \"none\", which does not suppress reasoning on "
                "this model family -- the model would reason to the token cap and those truncated "
                "rows would score as deliberate abstentions. Set REASONING_EFFORT (e.g. 'low')."
            )
        return
    if reasoning_effort is not None:
        raise ValueError(
            f"renderer {renderer_name!r} is controlled by `enable_thinking`, but "
            f"reasoning_effort={reasoning_effort!r} was configured. Pick the knob that "
            "matches the model family; sending both invites a silent mismatch."
        )
    expected = thinking_enabled_for(renderer_name)
    if expected != enable_thinking:
        raise ValueError(
            f"train/serve prompt mismatch: renderer {renderer_name!r} implies "
            f"enable_thinking={expected}, but eval is configured with "
            f"enable_thinking={enable_thinking}. Scores would measure the mismatch, "
            "not the model."
        )


def truncated_rows(results: list[dict]) -> int:
    """Rows that ran out of output budget, counted from saved `finish_reason`.

    A truncated row produces no parseable call, so it is byte-identical to a
    deliberate abstention: wrong on an AST category, *correct* on irrelevance.
    Unlike a generation failure there is no error to key on, so this is the only
    way to tell the two apart after a sweep. Non-zero on `irrelevance` means the
    score is partly a token-budget measurement.

    Returns 0 for sweeps recorded before `finish_reason` was saved.
    """
    return sum(1 for r in results if r.get("finish_reason") == "length")


def attribute_gain(
    category: str, rows: list[dict], base_results: list[dict], tuned_results: list[dict], answers: dict
) -> dict:
    """Split a score change into *emission* and *decision* components.

    A BFCL row can fail two very different ways, and the headline accuracy hides
    which one you fixed:

    - **Emission** -- the model never produced a parseable call, so there was
      nothing to check. Cause is formatting: wrong envelope, reasoning that ran
      to the token cap, or a refusal.
    - **Decision** -- a call came out, but the function or the arguments were
      wrong.

    Supervised fine-tuning on demonstrations from a *different* tool distribution
    reliably moves the first and barely touches the second: it teaches the shape
    of a tool call, not which call this particular benchmark wanted. Separating
    them tells you whether more of the same data will help, or whether you have
    hit the ceiling of what demonstrations can do and want a method that
    optimises the metric directly.

    `decision_acc` is deliberately conditioned on rows where **both** models
    emitted, so it is not contaminated by the emission change.
    """
    if category not in NON_LIVE_AST:
        raise ValueError(
            f"{category!r} has no ground truth to attribute against; "
            "the hallucination categories are scored structurally."
        )
    # lazy: `bfcl-eval` pulls a large dependency tree; only scoring needs it.
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

    language = _language_enum(category)
    by_row = {r["id"]: r for r in rows}

    def verdict(rid: str, rec: dict) -> bool | None:
        """True/False once a call exists; None when nothing parseable came out.

        A malformed call is an *emission* failure, not a decision failure: the
        official decoder parses the whole response at once, so one unparseable
        call means no parseable call came back for the row at all (see
        `_decode`). Scoring it `False` would book it as a wrong choice the model
        never got to make.
        """
        decoded, n_malformed = _decode(rec.get("result", []))
        if n_malformed or not decoded:
            return None
        try:
            out = ast_checker(
                by_row[rid]["function"], decoded, answers[rid], language, category, AST_NAME_STYLE_KEY
            )
            return bool(out.get("valid"))
        except Exception:  # noqa: BLE001 - a checker crash is a failed row
            return False

    base = {r["id"]: verdict(r["id"], r) for r in base_results}
    tuned = {r["id"]: verdict(r["id"], r) for r in tuned_results}
    ids = [r["id"] for r in rows if r["id"] in base and r["id"] in tuned]
    both = [i for i in ids if base[i] is not None and tuned[i] is not None]

    gain_emit = sum(1 for i in ids if base[i] is None and tuned[i] is True)
    loss_emit = sum(1 for i in ids if tuned[i] is None and base[i] is True)
    gain_dec = sum(1 for i in both if tuned[i] and not base[i])
    loss_dec = sum(1 for i in both if base[i] and not tuned[i])
    net = gain_emit - loss_emit + gain_dec - loss_dec

    return {
        "category": category,
        "total": len(ids),
        "base_emitted": sum(1 for i in ids if base[i] is not None),
        "tuned_emitted": sum(1 for i in ids if tuned[i] is not None),
        "both_emitted": len(both),
        "base_decision_acc": (sum(1 for i in both if base[i]) / len(both)) if both else 0.0,
        "tuned_decision_acc": (sum(1 for i in both if tuned[i]) / len(both)) if both else 0.0,
        "gain_emission": gain_emit,
        "loss_emission": loss_emit,
        "gain_decision": gain_dec,
        "loss_decision": loss_dec,
        "net_rows": net,
        "net_pp": 100 * net / len(ids) if ids else 0.0,
    }
