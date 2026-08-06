"""LangChain-free SQL tools for the serverless RL variant.

`sql_agent.py` builds a LangChain `create_agent`; that pulls in langgraph and is
tied to the managed-RFT rollout. The serverless loop
(`text2sql_serverless_rl.py`) runs its own multi-turn tool loop against the
Tinker sampler, so it needs plain functions + `ToolSpec`s instead:

  - get_database_schema(database_id)
  - look_up_evidence(query, database_id, k)
  - run_sql_against_database(sql_query, database_id)

`run_sql_against_database` appends the `sql_reward.RESULT_MARKER` (same as
`sql_agent.py`) so the reward can recover executed rows from the trajectory.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, List, Tuple

from sql_reward import RESULT_MARKER

HERE = Path(__file__).resolve().parent
DB_ROOT = Path(os.getenv("BIRD_DB_ROOT", str(HERE / "dev_databases")))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", str(HERE / "chromadb_text2sql")))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "sql_examples")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
MAX_RESULT_ROWS = int(os.getenv("MAX_RESULT_ROWS", "200"))


# --------------------------------------------------------------------------
# Tool specs (OpenAI/Tinker ToolSpec dicts) declared to the model
# --------------------------------------------------------------------------

TOOL_SPECS: List[dict] = [
    {
        "name": "get_database_schema",
        "description": "Get the schema (tables, columns) for a specific database. Call this FIRST.",
        "parameters": {
            "type": "object",
            "properties": {
                "database_id": {"type": "string", "description": "The database identifier, e.g. 'formula_1'."},
            },
            "required": ["database_id"],
        },
    },
    {
        "name": "look_up_evidence",
        "description": "Retrieve relevant SQL evidence/examples from a vector database. Call this after the schema.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to find relevant SQL examples for."},
                "database_id": {"type": "string", "description": "Optional database ID filter."},
                "k": {"type": "integer", "description": "Number of examples to retrieve (default 5)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "run_sql_against_database",
        "description": "Execute a SQL query against the specified database and return the rows.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql_query": {"type": "string", "description": "The SQL query to execute."},
                "database_id": {"type": "string", "description": "The database identifier, e.g. 'formula_1'."},
            },
            "required": ["sql_query", "database_id"],
        },
    },
]


def _db_path(database_id: str) -> Path:
    # Contain to DB_ROOT: a model-supplied database_id must be a bare name,
    # never a path or traversal, so it can't escape the dataset directory.
    if database_id != Path(database_id).name or database_id in ("", ".", ".."):
        raise ValueError(f"invalid database_id: {database_id!r}")
    return DB_ROOT / database_id / f"{database_id}.sqlite"


def describe_database(db_path: str | Path) -> str:
    """Compact schema description (tables + columns) for a SQLite db."""
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [r[0] for r in cur.fetchall() if not r[0].startswith("sqlite_")]
        lines: List[str] = []
        for t in tables:
            cur.execute(f'PRAGMA table_info("{t}")')
            cols = [f"{c[1]} ({c[2]})" for c in cur.fetchall()]
            lines.append(f"Table {t}:\n  " + "\n  ".join(cols))
        return "\n\n".join(lines) if lines else "(no tables)"
    finally:
        conn.close()


def make_embeddings():
    from langchain_fireworks import FireworksEmbeddings

    return FireworksEmbeddings(model=EMBED_MODEL)


def load_vector_store(embeddings=None):
    """Open the persisted Chroma evidence store built by `prepare_data.py`."""
    import chromadb
    from chromadb.config import Settings
    from langchain_chroma import Chroma

    embeddings = embeddings or make_embeddings()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    return Chroma(
        client=client,
        collection_name=CHROMA_COLLECTION,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


# --------------------------------------------------------------------------
# Tool implementations (return a plain string the model reads back)
# --------------------------------------------------------------------------

def get_database_schema(database_id: str) -> str:
    path = _db_path(database_id)
    if not path.exists():
        return f"Error: Database '{database_id}' not found at {path}"
    try:
        return f"Database Schema for '{database_id}':\n\n{describe_database(path)}"
    except Exception as e:  # noqa: BLE001
        return f"Error getting database schema: {e}"


def look_up_evidence(vector_store, query: str, database_id: str = "", k: int = 5) -> str:
    try:
        kwargs: dict[str, Any] = {"k": k}
        if database_id:
            kwargs["filter"] = {"db_id": database_id}
        results = vector_store.similarity_search_with_score(query, **kwargs)
        if not results:
            return f"No relevant evidence found for query: '{query}'"
        out = f"Found {len(results)} relevant examples:\n\n"
        for i, (doc, score) in enumerate(results, 1):
            out += f"Example {i} (similarity: {1 - score:.3f}):\n"
            out += f"Database: {doc.metadata.get('db_id')}\n"
            out += f"Evidence: {doc.page_content}\n"
            if doc.metadata.get("question"):
                out += f"Related Question: {doc.metadata['question']}\n"
            if doc.metadata.get("sql"):
                out += f"SQL Example: {doc.metadata['sql']}\n"
            out += "-" * 50 + "\n"
        return out
    except Exception as e:  # noqa: BLE001
        return f"Error looking up evidence: {e}"


def run_sql_against_database(sql_query: str, database_id: str) -> Tuple[str, list]:
    """Execute SQL; return (content_with_RESULT_MARKER, rows)."""
    path = _db_path(database_id)
    if not path.exists():
        return f"Error: Database '{database_id}' not found at {path}", []
    conn = sqlite3.connect(str(path))
    try:
        cur = conn.cursor()
        cur.execute(sql_query)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
    except Exception as e:  # noqa: BLE001
        return f"Error executing SQL query: {e}", []
    finally:
        conn.close()

    if not rows:
        preview = "Query executed successfully but returned no results."
    else:
        preview = f"Query executed successfully! Found {len(rows)} row(s):\n\n"
        if columns:
            preview += " | ".join(columns) + "\n" + "-" * len(" | ".join(columns)) + "\n"
        for r in rows[:10]:
            preview += " | ".join("NULL" if c is None else str(c) for c in r) + "\n"
        if len(rows) > 10:
            preview += f"\n... and {len(rows) - 10} more row(s)"

    capped = [list(r) for r in rows[:MAX_RESULT_ROWS]]
    return f"{preview}\n\n{RESULT_MARKER}{json.dumps(capped, default=str)}", capped


def execute_tool(name: str, arguments: dict, vector_store) -> str:
    """Dispatch a parsed tool call to its implementation; return the tool content."""
    try:
        if name == "get_database_schema":
            return get_database_schema(str(arguments.get("database_id", "")))
        if name == "look_up_evidence":
            return look_up_evidence(
                vector_store,
                str(arguments.get("query", "")),
                str(arguments.get("database_id", "") or ""),
                int(arguments.get("k", 5) or 5),
            )
        if name == "run_sql_against_database":
            content, _rows = run_sql_against_database(
                str(arguments.get("sql_query", "")), str(arguments.get("database_id", ""))
            )
            return content
        return f"Error: unknown tool '{name}'"
    except Exception as e:  # noqa: BLE001
        return f"Error executing tool '{name}': {e}"
