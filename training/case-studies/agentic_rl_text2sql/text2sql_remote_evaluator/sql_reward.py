"""Hybrid reward for the text-to-SQL agentic RL case study.

Reward = weighted blend of two components, each in [0, 1]:

1. ANSWER CORRECTNESS (deterministic): the agent's last executed SQL result set
   vs the pre-computed gold result set (order-insensitive multiset overlap).
2. TRAJECTORY (LLM-as-judge): does the agent follow the prescribed process ---
   look up the schema first, then retrieve evidence, THEN run SQL --- scored by
   a Fireworks judge model with grammar-constrained JSON output.

This module is intentionally dependency-light: it imports only `eval_protocol`,
`openai`, and the stdlib. It runs SERVER-SIDE in the Fireworks evaluator sandbox
and must NOT need the SQLite databases, Chroma, or LangChain --- the gold result
is cached in `input_metadata.dataset_info` at data-prep time, and the agent's
executed rows are embedded in the traced `run_sql_against_database` tool output
(see `RESULT_MARKER`). The rollout server (`sql_agent.py`) is the only producer
of that marker.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from eval_protocol.models import EvaluateResult, EvaluationRow, InputMetadata, Message

# Machine-readable results tag appended to every run_sql_against_database tool
# output by the rollout agent, so the (DB-less) evaluator can recover the exact
# rows the agent's query returned. Kept here so agent + reward never disagree.
RESULT_MARKER = "[[ROWS]]"


# --------------------------------------------------------------------------
# Dataset loading / split (mirrors tau2_reward.py helpers)
# --------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_train_holdout(
    rows: List[Dict[str, Any]],
    holdout_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Seeded shuffle then split into (train, holdout)."""
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n_holdout = max(1, int(len(shuffled) * holdout_ratio))
    return shuffled[n_holdout:], shuffled[:n_holdout]


def write_jsonl(rows: List[Dict[str, Any]], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


# --------------------------------------------------------------------------
# Result-set correctness (deterministic, DB-less)
# --------------------------------------------------------------------------

def _normalize_rows(rows: Any) -> Counter:
    """Turn a list-of-rows into an order-insensitive multiset of cell tuples.

    Numbers are rounded to 6 dp so 1 == 1.0; everything else is string-compared
    (whitespace-stripped). NULL/None -> the literal "NULL".
    """
    counter: Counter = Counter()
    if not isinstance(rows, list):
        return counter
    for row in rows:
        cells = row if isinstance(row, (list, tuple)) else [row]
        norm: List[str] = []
        for c in cells:
            if c is None:
                norm.append("NULL")
            elif isinstance(c, bool):
                norm.append(str(c))
            elif isinstance(c, (int, float)):
                norm.append(f"{round(float(c), 6):g}")
            else:
                norm.append(str(c).strip())
        counter[tuple(norm)] += 1
    return counter


def _extract_last_result_rows(messages: List[Message]) -> Any:
    """Recover the rows from the LAST run_sql result marker in the trajectory."""
    last: Any = None
    for m in messages:
        content = getattr(m, "content", None)
        if not isinstance(content, str) or RESULT_MARKER not in content:
            continue
        idx = content.rfind(RESULT_MARKER)
        payload = content[idx + len(RESULT_MARKER):].strip()
        try:
            last = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            continue
    return last


def answer_correctness(messages: List[Message], gold_result: Any) -> Tuple[float, str]:
    """Multiset overlap between the agent's executed rows and the gold rows.

    Exact match -> 1.0; otherwise Jaccard overlap (|intersection| / |union|)
    for a smoother GRPO signal on near-correct queries. No DB access needed.
    """
    gold = _normalize_rows(gold_result)
    pred_rows = _extract_last_result_rows(messages)
    if pred_rows is None:
        # Distinguish the two failure modes so the log is a real diagnostic:
        #   - the agent ran SQL but it errored (tool returned an error string), vs
        #   - the agent never called run_sql_against_database at all.
        ran_sql = any(
            isinstance(getattr(m, "content", None), str)
            and "Error executing SQL query" in m.content
            for m in messages
        )
        if ran_sql:
            return 0.0, "SQL errored (no rows returned)"
        return 0.0, "never ran run_sql_against_database"
    pred = _normalize_rows(pred_rows)
    if not gold and not pred:
        return 1.0, "both empty result sets"
    if gold == pred:
        return 1.0, "exact result-set match"
    inter = sum((gold & pred).values())
    union = sum((gold | pred).values())
    score = inter / union if union else 0.0
    return score, f"partial result-set match ({inter}/{union} rows)"


# --------------------------------------------------------------------------
# LLM-as-judge for trajectory adherence (async, grammar-constrained)
# --------------------------------------------------------------------------

DEFAULT_RUBRIC = [
    "The agent looked up the database schema (get_database_schema) before writing any SQL.",
    "The agent retrieved evidence (look_up_evidence) before finalizing its SQL answer.",
    "The agent called get_database_schema before look_up_evidence and run_sql_against_database (correct order).",
    "The agent used a valid database_id that matches the target database (no hallucinated db).",
    "The agent did not make redundant or clearly out-of-order tool calls.",
]

JUDGE_PROMPT = """You are grading whether a text-to-SQL agent followed the prescribed PROCESS during a conversation. You are grading the PROCESS/TRAJECTORY, not whether the final answer is numerically correct.

The agent is instructed to: (1) look up the database schema first, (2) then look up evidence/examples, (3) then write and run SQL.

TARGET DATABASE: {db_id}

CONVERSATION (agent turns, tool calls, and tool responses in order):
{trajectory}

CRITERIA:
{rubric}

For EACH criterion decide whether the agent satisfied it. Reply with ONLY a JSON object of this exact shape:
{{"results": [{{"criterion": "<repeat the criterion>", "met": true, "reasoning": "<one short sentence>"}}]}}"""


def _trajectory_to_text(messages: List[Message], max_chars: int = 6000) -> str:
    parts: List[str] = []
    for m in messages:
        role = getattr(m, "role", "") or ""
        content = getattr(m, "content", "") or ""
        # Strip the machine-readable result marker so the judge sees clean text.
        if isinstance(content, str) and RESULT_MARKER in content:
            content = content[: content.rfind(RESULT_MARKER)].rstrip()
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            calls = "; ".join(
                f"{tc.function.name}({tc.function.arguments})" for tc in tool_calls
            )
            content = f"{content} [tool_calls: {calls}]".strip()
        parts.append(f"{role}: {content}")
    text = "\n".join(parts)
    return text[:max_chars]  # keep the head (tool ORDER is what we grade)


async def llm_judge_trajectory(
    messages: List[Message],
    rubric: List[str],
    judge_model: str,
    api_key: str,
    db_id: str = "",
    base_url: str = "https://api.fireworks.ai/inference/v1",
) -> Tuple[float, str]:
    """Return (score in [0,1], reason). Score = fraction of rubric items met."""
    if not rubric:
        return 1.0, "no rubric"

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    prompt = JUDGE_PROMPT.format(
        db_id=db_id or "(unknown)",
        trajectory=_trajectory_to_text(messages),
        rubric=json.dumps(rubric, ensure_ascii=False, indent=2),
    )
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "TrajectoryJudgeResponse",
            "schema": {
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "criterion": {"type": "string"},
                                "met": {"type": "boolean"},
                                "reasoning": {"type": "string"},
                            },
                            "required": ["criterion", "met", "reasoning"],
                        },
                    }
                },
                "required": ["results"],
            },
        },
    }
    try:
        resp = await client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,  # reasoning judges spend tokens before the JSON
            temperature=0.0,
            response_format=response_format,
        )
        raw = resp.choices[0].message.content or ""
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                return 0.0, f"judge parse error: {raw[:120]}"
            obj = json.loads(match.group(0))
        results = obj.get("results", [])
        if not results:
            return 0.0, "judge returned no results"
        met = sum(1 for r in results if bool(r.get("met")))
        total = len(rubric)
        # Guard against a short/partial judge response inflating the score.
        score = met / total
        failed = [r.get("criterion", "?") for r in results if not r.get("met")]
        reason = "all criteria met" if met == total else f"{met}/{total} met; failed: {failed}"
        return score, reason
    except Exception as e:  # noqa: BLE001 - judge must never crash the rollout
        return 0.0, f"judge error: {type(e).__name__}: {str(e)[:120]}"


# --------------------------------------------------------------------------
# Hybrid reward
# --------------------------------------------------------------------------

async def compute_sql_reward(
    row: EvaluationRow,
    judge_model: str,
    api_key: str,
    use_llm_judge: bool = True,
    reward_mode: str = "weighted",
    weights: dict | None = None,
    verbose: bool = True,
) -> EvaluationRow:
    """Hybrid text-to-SQL reward: answer correctness + trajectory judge.

    Returns a scalar `EvaluateResult` on the row. On the managed RFT + remote
    rollout path the trainer captures exact tokens/logprobs at its gateway, so
    this function only needs to produce a per-trajectory score.
    """
    info = (row.input_metadata.dataset_info or {}) if row.input_metadata else {}
    gold_result = info.get("gold_result", [])
    db_id = info.get("db_id", "")
    rubric = info.get("trajectory_rubric") or DEFAULT_RUBRIC

    corr_score, corr_reason = answer_correctness(row.messages, gold_result)

    traj_score: float | None = None
    traj_reason = "trajectory judge skipped"
    if use_llm_judge:
        traj_score, traj_reason = await llm_judge_trajectory(
            row.messages, rubric, judge_model, api_key, db_id=db_id
        )

    w = weights or {"correctness": 0.6, "trajectory": 0.4}
    pairs = [(corr_score, w.get("correctness", 0.0))]
    if traj_score is not None:
        pairs.append((traj_score, w.get("trajectory", 0.0)))

    if reward_mode == "product":
        reward = 1.0
        for s, _ in pairs:
            reward *= s
    elif reward_mode == "mean":
        reward = sum(s for s, _ in pairs) / len(pairs)
    elif reward_mode == "weighted":
        wsum = sum(wt for _, wt in pairs)
        reward = (sum(s * wt for s, wt in pairs) / wsum) if wsum > 0 else 0.0
    else:
        raise ValueError(f"reward_mode must be 'product', 'mean', or 'weighted', got {reward_mode!r}")

    traj_str = f"trajectory={traj_score:.2f} ({traj_reason}) " if traj_score is not None else ""
    reason = (
        f"correctness={corr_score:.2f} ({corr_reason}) {traj_str}"
        f"[{reward_mode}] -> {reward:.3f}"
    )
    if verbose:
        rid = row.input_metadata.row_id if row.input_metadata else "?"
        print(f"[sql-reward][{rid}] {reason}")
    row.evaluation_result = EvaluateResult(
        score=reward,
        reason=reason,
        metrics={},
    )
    return row


# --------------------------------------------------------------------------
# EvaluationRow builder (used by prepare_data.py)
# --------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You are a helpful SQL assistant. You can:
1. Look up the database schema using the `get_database_schema` tool
2. Look up relevant SQL evidence/examples from a vector database using the `look_up_evidence` tool
3. Execute SQL queries against databases using the `run_sql_against_database` tool

ALWAYS follow this process when answering a question that requires SQL:
1. FIRST, call `get_database_schema` for the target database to understand its tables and columns.
2. THEN, call `look_up_evidence` to retrieve similar SQL examples that might help.
3. THEN, generate and execute the appropriate SQL query with `run_sql_against_database`.
4. Finally, give a clear natural-language answer to the user.

Be thorough in your reasoning and explain your approach.

Available databases:
{available_dbs}"""


def build_evaluation_row(
    row_id: str,
    question: str,
    db_id: str,
    gold_sql: str,
    gold_result: Any,
    available_dbs: List[str],
    rubric: List[str] | None = None,
) -> EvaluationRow:
    """Assemble one eval-protocol row: system + user messages + cached gold."""
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        available_dbs="\n".join(f"  - {d}" for d in available_dbs)
    )
    return EvaluationRow(
        messages=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=question),
        ],
        input_metadata=InputMetadata(
            row_id=row_id,
            dataset_info={
                "question": question,
                "db_id": db_id,
                "gold_sql": gold_sql,
                "gold_result": gold_result,
                "available_dbs": available_dbs,
                "trajectory_rubric": rubric or DEFAULT_RUBRIC,
            },
        ),
    )
