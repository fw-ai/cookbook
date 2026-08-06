"""Deterministic reward for the text-to-SQL agentic RL case study.

Reward = ANSWER CORRECTNESS in [0, 1]: the agent's last executed SQL result set
vs the pre-computed gold result set (order-insensitive multiset overlap; exact
match = 1.0, partial = Jaccard, no rows / errored / never ran = 0.0).

The gold result is cached in `input_metadata.dataset_info` at data-prep time, and
the agent's executed rows are embedded in the traced `run_sql_against_database`
tool output (see `RESULT_MARKER`), so scoring needs no DB access.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from eval_protocol.models import EvaluationRow, InputMetadata, Message

# Machine-readable results tag appended to every run_sql_against_database tool
# output by the rollout agent, so the (DB-less) evaluator can recover the exact
# rows the agent's query returned. Kept here so agent + reward never disagree.
RESULT_MARKER = "[[ROWS]]"


# --------------------------------------------------------------------------
# Dataset loading / split (mirrors tau2_reward.py helpers)
# --------------------------------------------------------------------------

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
            },
        ),
    )
