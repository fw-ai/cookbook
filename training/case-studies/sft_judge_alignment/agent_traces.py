"""Generate realistic agent traces with LangGraph + ChatFireworks.

The judge in this case study grades *agent traces*, so the traces have to fail
the way real agents fail: wrong joins, redundant queries, answers invented after
a tool errors. Hand-written templates cannot produce that distribution, so we run
an actual `create_react_agent` over a small SQLite store and vary the conditions
to spread quality across the 1-5 range.

Four tiers (see `TIERS`):

    strong      capable model, all tools, clean system prompt
    weak        smaller model, all tools -- fumbles joins and over-queries
    no_tools    capable model told to answer from memory -- confident hallucination
    flaky       capable model, but the SQL tool fails the first time it is called

Traces are serialized to the shape `render_trace` consumes:

    {"task_id", "task", "tier", "gold_answer", "trace": [{role, content, tool, args}]}
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

# ---------------------------------------------------------------------------
# The database the agent queries
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    country     TEXT NOT NULL,
    signup_year INTEGER NOT NULL,
    tier        TEXT NOT NULL
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    category   TEXT NOT NULL,
    unit_price REAL NOT NULL
);
CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_year  INTEGER NOT NULL,
    status      TEXT NOT NULL
);
CREATE TABLE order_items (
    order_id   INTEGER NOT NULL REFERENCES orders(order_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity   INTEGER NOT NULL
);
"""

_CATEGORIES = ["keyboard", "monitor", "cable", "headset", "dock"]
_COUNTRIES = ["US", "DE", "JP", "BR", "IN"]
_TIERS = ["free", "pro", "enterprise"]


def build_db(path: str | os.PathLike, seed: int = 7, *, snapshot_before: int | None = None) -> None:
    """Create the demo SQLite database at `path` (overwrites).

    `snapshot_before` drops every order from that year onward, producing a stale
    replica: same schema, same query results shape, quietly missing recent rows.
    """
    rng = random.Random(seed)
    path = Path(path)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.executescript(SCHEMA_SQL)

    customers = [
        (i, f"customer_{i}", rng.choice(_COUNTRIES), rng.choice([2021, 2022, 2023]), rng.choice(_TIERS))
        for i in range(1, 61)
    ]
    con.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", customers)

    products = [
        (i, f"{rng.choice(_CATEGORIES)}_{i}", rng.choice(_CATEGORIES), round(rng.uniform(9.5, 480.0), 2))
        for i in range(1, 31)
    ]
    con.executemany("INSERT INTO products VALUES (?,?,?,?)", products)

    orders, items, oid = [], [], 1
    for cid, *_ in customers:
        # At least 6 orders each: sparse data makes half the questions answer "0",
        # which collapses the judge's score distribution for reasons that have
        # nothing to do with agent quality.
        for _ in range(rng.randint(6, 14)):
            orders.append((oid, cid, rng.choice([2022, 2023, 2024]), rng.choice(["shipped", "shipped", "cancelled"])))
            for pid in rng.sample(range(1, 31), rng.randint(1, 3)):
                items.append((oid, pid, rng.randint(1, 5)))
            oid += 1
    con.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)
    con.executemany("INSERT INTO order_items VALUES (?,?,?)", items)
    if snapshot_before is not None:
        con.execute("DELETE FROM order_items WHERE order_id IN "
                    "(SELECT order_id FROM orders WHERE order_year >= ?)", (snapshot_before,))
        con.execute("DELETE FROM orders WHERE order_year >= ?", (snapshot_before,))
    con.commit()
    con.close()


SCHEMA_DOC = """Tables:
  customers(customer_id, name, country, signup_year, tier)
  products(product_id, name, category, unit_price)
  orders(order_id, customer_id, order_year, status)
  order_items(order_id, product_id, quantity)
"""


# ---------------------------------------------------------------------------
# Tasks: a question plus the SQL that produces the true answer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    task_id: str
    question: str
    gold_sql: str


def build_tasks() -> list[Task]:
    """A spread of lookup / aggregate / join / multi-hop questions."""
    tasks: list[Task] = []

    for cid in (3, 11, 24, 37, 52):
        tasks.append(Task(
            f"orders_{cid}",
            f"How many shipped orders did customer {cid} place in 2023?",
            f"SELECT COUNT(*) FROM orders WHERE customer_id={cid} AND order_year=2023 AND status='shipped'",
        ))
    for country in _COUNTRIES:
        tasks.append(Task(
            f"cust_{country}",
            f"How many customers are based in {country}?",
            f"SELECT COUNT(*) FROM customers WHERE country='{country}'",
        ))
    for cat in _CATEGORIES:
        tasks.append(Task(
            f"avgprice_{cat}",
            f"What is the average unit price of products in the '{cat}' category? Round to 2 decimals.",
            f"SELECT ROUND(AVG(unit_price),2) FROM products WHERE category='{cat}'",
        ))
    for year in (2022, 2023, 2024):
        tasks.append(Task(
            f"units_{year}",
            f"How many total units were sold across all shipped orders in {year}?",
            "SELECT COALESCE(SUM(oi.quantity),0) FROM order_items oi JOIN orders o ON o.order_id=oi.order_id "
            f"WHERE o.order_year={year} AND o.status='shipped'",
        ))
    for tier in _TIERS:
        tasks.append(Task(
            f"revenue_{tier}",
            f"What is the total revenue from shipped orders placed by '{tier}' tier customers? Round to 2 decimals.",
            "SELECT ROUND(COALESCE(SUM(oi.quantity*p.unit_price),0),2) FROM order_items oi "
            "JOIN orders o ON o.order_id=oi.order_id JOIN products p ON p.product_id=oi.product_id "
            f"JOIN customers c ON c.customer_id=o.customer_id WHERE c.tier='{tier}' AND o.status='shipped'",
        ))
    # Two-part answers. A single exactly-checkable number forces correctness to be
    # binary -- the agent either matches it or does not -- and the gold labels come
    # out as all 1s and 5s. Asking for a label *and* a quantity lets an agent be
    # half right, which is what populates the middle of the scale.
    for country in ("US", "DE", "JP"):
        tasks.append(Task(
            f"topcat_{country}",
            f"Which product category has the highest total units sold to customers in {country}, "
            "and how many units is that? Give both.",
            "SELECT p.category, SUM(oi.quantity) FROM order_items oi JOIN orders o ON o.order_id=oi.order_id "
            "JOIN products p ON p.product_id=oi.product_id JOIN customers c ON c.customer_id=o.customer_id "
            f"WHERE c.country='{country}' GROUP BY p.category ORDER BY SUM(oi.quantity) DESC LIMIT 1",
        ))
    for tier in _TIERS:
        tasks.append(Task(
            f"bigspender_{tier}",
            f"Among '{tier}' tier customers, which customer_id spent the most on shipped orders, "
            "and what did they spend? Round the amount to 2 decimals.",
            "SELECT c.customer_id, ROUND(SUM(oi.quantity*p.unit_price),2) FROM order_items oi "
            "JOIN orders o ON o.order_id=oi.order_id JOIN products p ON p.product_id=oi.product_id "
            f"JOIN customers c ON c.customer_id=o.customer_id WHERE c.tier='{tier}' AND o.status='shipped' "
            "GROUP BY c.customer_id ORDER BY 2 DESC LIMIT 1",
        ))
    for year in (2022, 2023):
        tasks.append(Task(
            f"cancelrate_{year}",
            f"What percentage of {year} orders were cancelled? Round to 1 decimal.",
            "SELECT ROUND(100.0*SUM(status='cancelled')/COUNT(*),1) FROM orders "
            f"WHERE order_year={year}",
        ))
    return tasks


def stale_path(db_path: str | os.PathLike) -> Path:
    p = Path(db_path)
    return p.with_suffix(STALE_SUFFIX + p.suffix)


def gold_answer(db_path: str | os.PathLike, task: Task) -> str:
    con = sqlite3.connect(db_path)
    try:
        row = con.execute(task.gold_sql).fetchone()
    finally:
        con.close()
    if row is None:
        return ""
    return " | ".join("" if v is None else str(v) for v in row)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

MAX_ROWS = 25

# Per-HTTP-call timeout, and a wall-clock budget for a whole trace (a ReAct loop
# makes several calls, so the trace deadline has to be a multiple of the call one).
REQUEST_TIMEOUT_S = 90
TRACE_TIMEOUT_S = 300


def make_tools(db_path: str | os.PathLike, *, flaky: bool = False) -> list[Callable[..., Any]]:
    """Build the agent's tools.

    `flaky=True` makes the first `run_sql` call of a run return a transient
    error. That is the tier that reveals whether an agent retries or gives up
    and invents an answer -- the exact behaviour we want the judge to catch.
    """
    from langchain_core.tools import tool

    state = {"sql_calls": 0}
    lock = threading.Lock()

    @tool
    def run_sql(query: str) -> str:
        """Run a read-only SQL query against the store and return rows as text."""
        with lock:
            state["sql_calls"] += 1
            first = state["sql_calls"] == 1
        if flaky and first:
            return "ERROR: connection reset by peer (transient). Retry the query."
        stripped = query.strip().rstrip(";").lstrip("(").strip()
        if not stripped.lower().startswith(("select", "with")):
            return "ERROR: only SELECT/WITH queries are allowed."
        con = sqlite3.connect(db_path)
        try:
            cur = con.execute(query)
            rows = cur.fetchmany(MAX_ROWS + 1)
            cols = [d[0] for d in cur.description] if cur.description else []
        except Exception as exc:
            return f"ERROR: {exc}"
        finally:
            con.close()
        if not rows:
            return "(0 rows)"
        truncated = len(rows) > MAX_ROWS
        body = "\n".join(" | ".join("" if v is None else str(v) for v in r) for r in rows[:MAX_ROWS])
        header = " | ".join(cols)
        suffix = f"\n... (truncated at {MAX_ROWS} rows)" if truncated else ""
        return f"{header}\n{body}{suffix}"

    @tool
    def describe_schema() -> str:
        """Return the database schema (table and column names)."""
        return SCHEMA_DOC

    @tool
    def calculator(expression: str) -> str:
        """Evaluate a arithmetic expression, e.g. '100.0 * 7 / 43'."""
        allowed = set("0123456789.+-*/() ")
        if not set(expression) <= allowed:
            return "ERROR: only arithmetic characters are allowed."
        try:
            return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 - digits/operators only
        except Exception as exc:
            return f"ERROR: {exc}"

    return [run_sql, describe_schema, calculator]


# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------

BASE_PROMPT = (
    "You are a data analyst agent answering questions about an e-commerce store.\n"
    "Use the tools to query the database. Call describe_schema if you are unsure of the columns.\n"
    "Write targeted SQL -- do not select whole tables and filter by hand.\n"
    "Finish with a short, direct answer stating the number or name you found.\n\n"
    f"{SCHEMA_DOC}"
)

# A stale schema, the kind that survives in a prompt after a migration: `orders`
# lost its `status`, `customers.tier` is now `plan`, `order_items` renamed. The
# agent writes confident SQL against columns that no longer exist, has to recover
# from real errors, and often answers from a half-corrected query. This produces
# genuine mid-range traces without depending on finding a weak enough model.
STALE_SCHEMA_DOC = """Tables:
  customers(customer_id, name, region, signup_date, plan)
  products(product_id, title, category, price)
  orders(order_id, customer_id, order_date, state, total)
  line_items(order_id, product_id, qty)
"""

STALE_PROMPT = (
    "You are a data analyst agent answering questions about an e-commerce store.\n"
    "Use the tools to query the database. Write targeted SQL and answer directly.\n"
    "The schema below is authoritative; trust it.\n\n"
    f"{STALE_SCHEMA_DOC}"
)

# No tools at all -- the retrieval layer is down. The interesting question for the
# judge is whether the agent says "I cannot check" or invents a plausible number.
MEMORY_PROMPT = (
    "You are a data analyst agent answering questions about an e-commerce store.\n"
    "You have no database access right now. Answer the question as best you can.\n\n"
    f"{SCHEMA_DOC}"
)


@dataclass(frozen=True)
class Tier:
    name: str
    model: str
    prompt: str
    flaky: bool = False
    tools: bool = True
    max_steps: int = 8
    temperature: float = 0.3
    # Query a stale replica instead of the live store. The agent's method is fine;
    # the number it returns is not.
    stale_data: bool = False
    # Share of the trace budget. See build_tiers for why these are not all 1.
    weight: int = 1


def build_tiers(strong_model: str, weak_model: str) -> list[Tier]:
    """The tier mix, deliberately not uniform.

    Only some tiers produce mid-scale labels. `strong` is almost always a 5 and
    `no_tools` is almost always a 1, so sampling all five tiers equally spends most
    of the budget on cases nobody disagrees about and leaves the middle of the scale
    empty -- which makes every judge look good.

    Weighting toward the contested tiers is the same reason real eval sets are
    stratified toward the decision boundary instead of mirroring production traffic.
    Keep some `strong` and `no_tools`: without the easy ends the scale has no anchors.
    """
    return [
        Tier("strong", strong_model, BASE_PROMPT, weight=2),
        Tier("stale_schema", weak_model, STALE_PROMPT, max_steps=6, temperature=0.7, weight=3),
        Tier("stale_data", strong_model, BASE_PROMPT, stale_data=True, weight=4),
        Tier("no_tools", strong_model, MEMORY_PROMPT, tools=False, max_steps=2, weight=1),
        Tier("flaky", weak_model, BASE_PROMPT, flaky=True, max_steps=4, temperature=0.7, weight=2),
    ]


TIERS = ("strong", "stale_schema", "stale_data", "no_tools", "flaky")

# Suffix appended to the stale replica's filename.
STALE_SUFFIX = ".stale"
SNAPSHOT_YEAR = 2024   # the stale replica is missing everything from this year on


# ---------------------------------------------------------------------------
# Running the agent
# ---------------------------------------------------------------------------

def serialize(messages) -> list[dict[str, Any]]:
    """LangChain message objects -> the flat trace dicts the judge prompt renders.

    Tool calls are attached to the assistant turn that issued them so a trace
    reads as "assistant said X and called Y", which is how a human reviewer
    thinks about it. The system prompt is dropped: the judge grades behaviour,
    and leaving it in would leak the tier (the sabotaged prompt is recognisable).
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        mtype = getattr(m, "type", None)
        if mtype == "system":
            continue
        if mtype == "human":
            out.append({"role": "user", "content": m.content or ""})
        elif mtype == "ai":
            calls = getattr(m, "tool_calls", None) or []
            content = m.content if isinstance(m.content, str) else json.dumps(m.content)
            if calls:
                for call in calls:
                    out.append({
                        "role": "assistant",
                        "content": content,
                        "tool": call.get("name", ""),
                        "args": json.dumps(call.get("args", {}), ensure_ascii=False),
                    })
                    content = ""  # only the first row carries the assistant's text
            else:
                out.append({"role": "assistant", "content": content})
        elif mtype == "tool":
            text = m.content if isinstance(m.content, str) else json.dumps(m.content)
            out.append({"role": "tool", "content": text})
    return out


def render_trace(trace: list[dict[str, Any]]) -> str:
    """Flat trace dicts -> the plain-text transcript shown to judges and annotators."""
    lines = []
    for m in trace:
        role = m["role"]
        if role == "assistant" and m.get("tool"):
            prefix = f"ASSISTANT: {m['content']}\n  " if m.get("content") else "  "
            lines.append(f"{prefix}-> CALL {m['tool']}({m['args']})")
        elif role == "tool":
            lines.append(f"TOOL RESULT: {m['content']}")
        else:
            lines.append(f"{role.upper()}: {m['content']}")
    return "\n".join(lines)


def final_answer(trace: list[dict[str, Any]]) -> str:
    for m in reversed(trace):
        if m["role"] == "assistant" and m.get("content") and not m.get("tool"):
            return m["content"]
    return ""


def run_one(task: Task, tier: Tier, db_path: str | os.PathLike, *, api_key: str | None = None) -> dict[str, Any]:
    """Run a single task under a single tier and return the serialized trace."""
    from langchain_fireworks import ChatFireworks
    from langgraph.prebuilt import create_react_agent

    # `request_timeout` defaults to None, which means a stalled request blocks forever
    # and takes the whole batch with it (ThreadPoolExecutor.map waits on every worker).
    # Bound each HTTP call and keep retries low; a slow trace is worth abandoning.
    llm = ChatFireworks(
        model=tier.model,
        temperature=tier.temperature,
        max_tokens=1024,
        request_timeout=REQUEST_TIMEOUT_S,
        max_retries=1,
        api_key=api_key or os.environ["FIREWORKS_API_KEY"],
    )
    # The gold answer always comes from the live store, even when the agent is
    # pointed at the stale replica -- that mismatch is the whole point of the tier.
    query_db = stale_path(db_path) if tier.stale_data else db_path
    tools = make_tools(query_db, flaky=tier.flaky) if tier.tools else []
    agent = create_react_agent(llm, tools, prompt=tier.prompt)
    record: dict[str, Any] = {
        "task_id": task.task_id,
        "task": task.question,
        "tier": tier.name,
        "gold_answer": gold_answer(db_path, task),
    }
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": task.question}]},
            {"recursion_limit": tier.max_steps * 2},
        )
        record["trace"] = serialize(result["messages"])
    except Exception as exc:
        # A blown step budget or a provider error is itself a trace worth judging:
        # keep it, tagged, rather than silently biasing the set toward completions.
        record["trace"] = [
            {"role": "user", "content": task.question},
            {"role": "assistant", "content": f"[run aborted: {type(exc).__name__}: {exc}]"},
        ]
        record["aborted"] = True
    record["answer"] = final_answer(record["trace"])
    return record


def generate(
    out_path: str | os.PathLike,
    db_path: str | os.PathLike,
    *,
    strong_model: str,
    weak_model: str,
    n_traces: int = 200,
    concurrency: int = 8,
    seed: int = 0,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Run tasks x tiers until `n_traces` traces exist, and write them to JSONL."""
    build_db(db_path)
    build_db(stale_path(db_path), snapshot_before=SNAPSHOT_YEAR)
    tasks = build_tasks()
    tiers = build_tiers(strong_model, weak_model)

    pairs = _stratify(tasks, tiers, db_path, n_traces, seed)

    done = {"n": 0}
    lock = threading.Lock()

    def work(pair):
        rec = run_one(pair[0], pair[1], db_path)
        if progress:
            with lock:
                done["n"] += 1
                if done["n"] % 10 == 0 or done["n"] == len(pairs):
                    print(f"  traces {done['n']}/{len(pairs)}", flush=True)
        return rec

    # Every trace gets a hard wall-clock deadline. A single wedged request must not
    # be able to hold the run open indefinitely, so timed-out traces are recorded as
    # aborted (they are legitimate traces to judge) and the batch moves on.
    pool = ThreadPoolExecutor(max_workers=concurrency)
    try:
        futures = {pool.submit(work, p): p for p in pairs}
        # Traces run `concurrency` at a time, so the budget is the per-trace deadline
        # times the number of waves, plus slack -- not a single trace's worth.
        waves = -(-len(pairs) // max(1, concurrency))
        deadline = time.monotonic() + TRACE_TIMEOUT_S * waves + 120
        records = []
        for fut, (task, tier) in futures.items():
            try:
                records.append(fut.result(timeout=max(5.0, deadline - time.monotonic())))
            except FuturesTimeout:
                fut.cancel()
                print(f"  [timeout] {task.task_id} / {tier.name} -- abandoned", flush=True)
                records.append({
                    "task_id": task.task_id, "task": task.question, "tier": tier.name,
                    "gold_answer": gold_answer(db_path, task), "aborted": True, "answer": "",
                    "trace": [{"role": "user", "content": task.question},
                              {"role": "assistant", "content": "[run abandoned: timed out]"}],
                })
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records


def diverging_tasks(tasks: Sequence[Task], db_path: str | os.PathLike) -> list[Task]:
    """Tasks whose answer actually differs between the live store and the stale replica.

    Only about a third of the questions touch the orders tables, so pairing the
    `stale_data` tier with a random task usually produces a *correct* answer and
    wastes the tier. Checking which tasks diverge is two SQL queries each.
    """
    stale = stale_path(db_path)
    return [t for t in tasks if gold_answer(db_path, t) != gold_answer(stale, t)]


def _stratify(
    tasks: Sequence[Task],
    tiers: Sequence[Tier],
    db_path: str | os.PathLike,
    n_traces: int,
    seed: int,
) -> list[tuple[Task, Tier]]:
    """Allocate the trace budget across tiers by weight, on tasks where each can bite.

    A uniform shuffle over `tasks x tiers` looks fair and is not: the tiers that
    generate mid-scale labels are exactly the ones with extra requirements, so
    random pairing dilutes them and the gold set collapses toward 1s and 5s.
    """
    rng = random.Random(seed)
    diverging = diverging_tasks(tasks, db_path)
    total_weight = sum(t.weight for t in tiers)

    pairs: list[tuple[Task, Tier]] = []
    for tier in tiers:
        share = max(1, round(n_traces * tier.weight / total_weight))
        # stale_data only earns its keep on tasks the stale replica gets wrong.
        pool = list(diverging if (tier.stale_data and diverging) else tasks)
        rng.shuffle(pool)
        pairs.extend((pool[i % len(pool)], tier) for i in range(share))

    rng.shuffle(pairs)
    return pairs[:n_traces]


def load(path: str | os.PathLike) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
