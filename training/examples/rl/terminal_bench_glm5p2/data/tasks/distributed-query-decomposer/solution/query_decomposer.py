"""
Distributed SQL Query Decomposer for DuckDB.

Transforms aggregate SQL queries into a (worker_query, summary_query) pair
for correct parallel execution across horizontally-partitioned data shards.

Decomposition rules:
  SUM(x)        -> worker: SUM(x)              summary: SUM(partial)
  COUNT(x)/(*) -> worker: COUNT(x)/(*)        summary: SUM(partial)
  MIN(x)        -> worker: MIN(x)              summary: MIN(partial)
  MAX(x)        -> worker: MAX(x)              summary: MAX(partial)
  AVG(x)        -> worker: SUM(x), COUNT(x)   summary: SUM(s)/SUM(c)
  VAR_POP(x)    -> worker: SUM(x*x), SUM(x), COUNT(x)
                   summary: SUM(ss)/SUM(c) - (SUM(s)/SUM(c))^2
  STDDEV_POP(x) -> same as VAR_POP wrapped in SQRT
"""

import sqlglot
from sqlglot import exp


def decompose_query(sql: str, source_table: str) -> dict:
    """Decompose a SQL aggregate query into worker and summary queries.

    Args:
        sql: Original SQL SELECT query with aggregates (DuckDB dialect).
        source_table: Name of the table being horizontally partitioned.

    Returns:
        dict with "worker_query" and "summary_query" string values.
    """
    tree = sqlglot.parse_one(sql, dialect="duckdb")
    dc = _Decomposer()

    select_node = tree.find(exp.Select)
    where_node = tree.find(exp.Where)
    group_node = tree.find(exp.Group)
    having_node = tree.find(exp.Having)
    order_node = tree.find(exp.Order)
    limit_node = tree.find(exp.Limit)

    # Decompose SELECT items
    for item in select_node.expressions:
        dc.process_select_item(item)

    # Decompose aggregates that appear only in HAVING
    if having_node:
        dc.process_having_aggs(having_node)

    worker_sql = dc.build_worker(source_table, where_node, group_node)
    summary_sql = dc.build_summary(group_node, having_node, order_node, limit_node)

    return {"worker_query": worker_sql, "summary_query": summary_sql}


# ---------------------------------------------------------------------------

_DIALECT = "duckdb"

_AGG_ANON_NAMES = frozenset({
    "VAR_POP", "VARIANCE_POP", "STDDEV_POP",
})


class _Decomposer:
    """Stateful helper that accumulates worker/summary column lists."""

    def __init__(self):
        self.worker_cols: list[str] = []
        self.summary_cols: list[str] = []
        self.agg_map: dict[str, str] = {}   # agg_sql -> summary_expr
        self._seq = 0

    # -- id generation -------------------------------------------------------

    def _next_id(self) -> int:
        self._seq += 1
        return self._seq

    # -- aggregate detection -------------------------------------------------

    @staticmethod
    def _find_aggs(node):
        """Return all aggregate-function nodes, including Anonymous ones."""
        result = list(node.find_all(exp.AggFunc))
        for anon in node.find_all(exp.Anonymous):
            if anon.name.upper() in _AGG_ANON_NAMES:
                result.append(anon)
        return result

    @staticmethod
    def _agg_kind(node) -> str:
        sql_upper = node.sql(dialect=_DIALECT).upper().strip()
        for tag in ("VAR_POP", "VARIANCE_POP", "STDDEV_POP",
                     "SUM", "COUNT", "MIN", "MAX", "AVG"):
            if sql_upper.startswith(tag + "("):
                return tag.replace("VARIANCE_POP", "VAR_POP")
        name = type(node).__name__
        _map = {
            "Sum": "SUM", "Count": "COUNT", "Min": "MIN", "Max": "MAX",
            "Avg": "AVG", "VariancePop": "VAR_POP", "StddevPop": "STDDEV_POP",
            "Variance": "VAR_POP", "Stddev": "STDDEV_POP",
        }
        if name in _map:
            return _map[name]
        raise ValueError(f"Unsupported aggregate function: {sql_upper}")

    @staticmethod
    def _agg_arg(node) -> str | None:
        """Return the SQL text of the aggregate's argument, or None for COUNT(*)."""
        if isinstance(node, exp.Count):
            if node.this is None or isinstance(node.this, exp.Star):
                return None
            return node.this.sql(dialect=_DIALECT)
        if isinstance(node, exp.Anonymous):
            if node.expressions:
                return node.expressions[0].sql(dialect=_DIALECT)
            return None
        if hasattr(node, "this") and node.this is not None:
            return node.this.sql(dialect=_DIALECT)
        return None

    # -- single-aggregate decomposition --------------------------------------

    def _decompose_one(self, node):
        """Return (worker_col_defs: list[str], summary_expr: str)."""
        n = self._next_id()
        k = self._agg_kind(node)
        a = self._agg_arg(node)

        if k == "SUM":
            w = f"_ws{n}"
            return [f"SUM({a}) AS {w}"], f"SUM({w})"

        if k == "COUNT":
            w = f"_wc{n}"
            src = "COUNT(*)" if a is None else f"COUNT({a})"
            return [f"{src} AS {w}"], f"SUM({w})"

        if k == "MIN":
            w = f"_wn{n}"
            return [f"MIN({a}) AS {w}"], f"MIN({w})"

        if k == "MAX":
            w = f"_wx{n}"
            return [f"MAX({a}) AS {w}"], f"MAX({w})"

        if k == "AVG":
            ws, wc = f"_was{n}", f"_wac{n}"
            return (
                [f"CAST(SUM({a}) AS DOUBLE) AS {ws}",
                 f"COUNT({a}) AS {wc}"],
                f"(SUM({ws}) / SUM({wc}))",
            )

        if k == "VAR_POP":
            wss, ws, wc = f"_wvss{n}", f"_wvs{n}", f"_wvc{n}"
            return (
                [f"CAST(SUM(({a}) * ({a})) AS DOUBLE) AS {wss}",
                 f"CAST(SUM({a}) AS DOUBLE) AS {ws}",
                 f"COUNT({a}) AS {wc}"],
                f"((SUM({wss}) / SUM({wc})) - POWER(SUM({ws}) / SUM({wc}), 2))",
            )

        if k == "STDDEV_POP":
            wss, ws, wc = f"_wdss{n}", f"_wds{n}", f"_wdc{n}"
            return (
                [f"CAST(SUM(({a}) * ({a})) AS DOUBLE) AS {wss}",
                 f"CAST(SUM({a}) AS DOUBLE) AS {ws}",
                 f"COUNT({a}) AS {wc}"],
                f"SQRT((SUM({wss}) / SUM({wc})) - POWER(SUM({ws}) / SUM({wc}), 2))",
            )

        raise ValueError(f"Unhandled aggregate kind: {k}")

    def _ensure_agg(self, node) -> str:
        """Decompose *node* if not yet seen; return its summary expression."""
        key = node.sql(dialect=_DIALECT)
        if key not in self.agg_map:
            wcols, sexpr = self._decompose_one(node)
            self.worker_cols.extend(wcols)
            self.agg_map[key] = sexpr
        return self.agg_map[key]

    # -- SELECT / HAVING processing ------------------------------------------

    def process_select_item(self, item):
        if isinstance(item, exp.Alias):
            alias, core = item.alias, item.this
        else:
            alias, core = None, item

        aggs = self._find_aggs(core)

        if not aggs:
            # Non-aggregate expression — pass through unchanged
            self.worker_cols.append(item.sql(dialect=_DIALECT))
            self.summary_cols.append(alias if alias else core.sql(dialect=_DIALECT))
        else:
            summary_expr = core.sql(dialect=_DIALECT)
            for ag in aggs:
                ag_sql = ag.sql(dialect=_DIALECT)
                s = self._ensure_agg(ag)
                summary_expr = summary_expr.replace(ag_sql, f"({s})", 1)
            self.summary_cols.append(
                f"{summary_expr} AS {alias}" if alias else summary_expr
            )

    def process_having_aggs(self, having_node):
        """Ensure any aggregates in HAVING are decomposed (may add worker cols)."""
        for ag in self._find_aggs(having_node):
            self._ensure_agg(ag)

    # -- query assembly ------------------------------------------------------

    def build_worker(self, table, where_node, group_node) -> str:
        parts = [f"SELECT {', '.join(self.worker_cols)}", f"FROM {table}"]
        if where_node:
            parts.append(where_node.sql(dialect=_DIALECT))
        if group_node:
            parts.append(group_node.sql(dialect=_DIALECT))
        return " ".join(parts)

    def build_summary(self, group_node, having_node, order_node, limit_node) -> str:
        parts = [f"SELECT {', '.join(self.summary_cols)}", "FROM combined_results"]
        if group_node:
            parts.append(group_node.sql(dialect=_DIALECT))
        if having_node:
            h = having_node.sql(dialect=_DIALECT)
            # Replace longest matches first to avoid partial substitution
            for orig, repl in sorted(self.agg_map.items(), key=lambda x: -len(x[0])):
                h = h.replace(orig, f"({repl})")
            parts.append(h)
        if order_node:
            parts.append(order_node.sql(dialect=_DIALECT))
        if limit_node:
            parts.append(limit_node.sql(dialect=_DIALECT))
        return " ".join(parts)
