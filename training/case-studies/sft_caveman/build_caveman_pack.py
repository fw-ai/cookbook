#!/usr/bin/env python3
"""Build the scaled-up Caveman SFT pack by self-distillation.

The Caveman behavior already exists in GLM 5.3 Flash when it is prompted with
the skill file. This pipeline copies that behavior into training data so an
adapter can carry it as the default:

  tasks     author a few hundred small, self-contained fixtures (a question
            plus two to four facts, with the literals an answer must keep),
            across the four task families the pilot used, and split them by
            fixture group into train / dev / test.
  targets   the TEACHER (base model + skill prompt) answers each training
            fixture twice; a dev/test fixture is never sent to the teacher.
  filter    keep a target only if it passes the checks the human review of the
            pilot turned into rules: every required literal present, no
            number or name that is absent from the input, nothing answered
            that was not asked, exact format honored. A judge enforces the
            semantic parts; the literal checks are plain code. The shorter
            passing target wins.
  pack      write pack/caveman_train.jsonl (student prompt + accepted target),
            pack/caveman_dev.jsonl and pack/caveman_test.jsonl (prompts only,
            with required literals, for evaluation), and pack/manifest.json.

Usage:
    export FIREWORKS_API_KEY=...
    python build_caveman_pack.py all          # or: tasks | targets | filter | pack
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
# CAVEMAN_DATA_DIR lets a scaled rebuild live beside the frozen pack: data-v2/ holds its tasks, targets,
# scores, costs ledger and a pack/ subfolder. Default is the original layout (data/ and pack/).
_DD = os.environ.get("CAVEMAN_DATA_DIR")
DATA = Path(_DD) if _DD else HERE / "data"
PACK = DATA / "pack" if _DD else HERE / "pack"
COSTS = DATA / "costs.jsonl"
import cost_ledger  # noqa: E402
API_BASE = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai").rstrip("/")
CHAT_URL = f"{API_BASE}/inference/v1/chat/completions"

BASE_MODEL = "accounts/fireworks/models/glm-5p3-flash"      # teacher = base model + skill prompt
AUTHOR_MODEL = "accounts/fireworks/models/glm-5p3-flash"    # writes the fixtures
JUDGE_MODEL = "accounts/fireworks/models/deepseek-v4p1-flash"   # a different family checks the targets

STUDENT_SYSTEM = "Answer the user using the supplied task information. Do not invent missing facts."
SKILL_SUFFIX = ("\n\nFor this response use English full mode. Honor the skill’s clarity exceptions and the "
                "user’s explicit output requirements.")

N_TASKS = int(os.environ.get("CAVEMAN_N_TASKS", 320))
SEED = int(os.environ.get("CAVEMAN_SEED", 29))
FAMILIES = {
    # family: (share, authoring instruction)
    "technical": (0.55, "a grounded technical question whose answer follows from the facts by one or two steps "
                        "(limits, defaults, boundaries, arithmetic, precedence, paths, versions)"),
    "format": (0.20, "an instruction to return ONLY a specific exact artifact (a CSV, a one-line YAML, a regex, "
                     "a SQL statement, a JSON object) built from the facts, with the exact expected output derivable"),
    "clarity": (0.15, "a question that explicitly demands a clarity exception: full sentences, no fragments, no "
                      "pronouns, or one normal-English sentence for a report"),
    "cannot_tell": (0.10, "a question the facts deliberately cannot answer, so the right answer says so and names "
                          "what is missing, without inventing a procedure"),
}
if os.environ.get("CAVEMAN_FAMILIES") == "v2":
    # Scaled build: two more families so the adapter generalizes past one-shot technical lookups.
    FAMILIES = {
        "technical": (0.40, FAMILIES["technical"][1]),
        "format": (0.15, FAMILIES["format"][1]),
        "clarity": (0.12, FAMILIES["clarity"][1]),
        "cannot_tell": (0.10, FAMILIES["cannot_tell"][1]),
        "everyday": (0.13, "a practical, non-technical question (cooking, travel, household, fitness, money, scheduling) "
                           "whose answer follows from the facts; the facts describe a concrete personal situation"),
        "multi_part": (0.10, "a question with two or three explicit sub-questions, each answered from a different fact "
                             "(4 facts), where the answer must address every part in order"),
    }
TARGETS_PER_TASK = int(os.environ.get("CAVEMAN_TARGETS_PER_TASK", 2))
TEACHER_TEMPERATURE = 0.7
TEACHER_MAX_TOKENS = int(os.environ.get("CAVEMAN_TEACHER_MAX_TOKENS", 1024))
# The served adapter is called at effort low; the teacher may think at max to write better targets (the
# trace is not trained, only the visible answer). Past finding: max-effort teachers give cleaner targets.
TEACHER_EFFORT = os.environ.get("CAVEMAN_TEACHER_EFFORT", "low")
JUDGE_MIN = 0.8
CONCURRENCY = int(os.environ.get("CAVEMAN_CONCURRENCY", 8))


# ----------------------------------------------------------------------------
def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


# ----------------------------------------------------------------------------
# Cache identity. Targets and scores are cached on disk between runs. A cache
# row is reused only when the INPUT it was made for still has the same content
# (ids are sequential and change when the inputs stage reruns) and the teacher
# or judge configuration that produced it is unchanged. Stale rows are dropped
# from the file on load, so a rebuilt input never inherits an old answer.
FILTER_VERSION = 2   # bump when literal_checks / judge prompt semantics change


def fingerprint(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def load_cache(path: Path, current: dict[str, str], id_field: str, cfg_field: str, cfg_fp,
               drop_status: str | None = None) -> list[dict]:
    """Rows of ``path`` whose input (by ``id_field``) still has fingerprint ``current[id]`` and whose
    ``cfg_field`` equals ``cfg_fp`` (a string, or a dict id -> expected fingerprint when the config differs per
    input, e.g. a teacher hint for one family); the file is rewritten with only those rows."""
    rows = read_jsonl(path)
    want = (lambda r: cfg_fp.get(r[id_field])) if isinstance(cfg_fp, dict) else (lambda r: cfg_fp)
    kept = [r for r in rows if current.get(r[id_field]) == r.get("input_fp") and r.get(cfg_field) == want(r)
            and (drop_status is None or r.get("status") != drop_status)]
    if len(kept) != len(rows):
        print(f"  cache {path.name}: kept {len(kept)}/{len(rows)} rows (rest stale: input changed, config changed, or failed)")
        write_jsonl(path, kept)
    return kept


def require_tasks() -> list[dict]:
    """The verified task file is the root of every later stage. Refuse to run without it rather than
    rewriting the frozen pack from nothing."""
    tasks = read_jsonl(DATA / "caveman_tasks.jsonl")
    if not tasks:
        raise SystemExit(f"{DATA / 'caveman_tasks.jsonl'} is missing or empty: run the `tasks` stage first (or restore data/)")
    return tasks


def input_fp(t: dict) -> str:
    return fingerprint(t["question"], t["facts"], t["expected"], t["required_literals"], t["student_messages"])


TEACHER_FAMILY_HINT_VERSION = {"cannot_tell": 1}   # bump a family to regenerate only its targets


def teacher_fp(family: str | None = None) -> str:
    v = TEACHER_FAMILY_HINT_VERSION.get(family or "", 0)
    base = (BASE_MODEL, TEACHER_TEMPERATURE, TEACHER_MAX_TOKENS, TEACHER_EFFORT, skill_body(), SKILL_SUFFIX)
    return fingerprint(*base) if v == 0 else fingerprint(*base, family, v)


def judge_fp() -> str:
    return fingerprint(JUDGE_MODEL, JUDGE_MIN, FILTER_VERSION, JUDGE_SYS, inspect.getsource(judge), inspect.getsource(literal_checks), inspect.getsource(effective_literals))


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def api_key() -> str:
    k = os.environ.get("FIREWORKS_API_KEY")
    if not k:
        raise SystemExit("set FIREWORKS_API_KEY")
    return k


_TIMEOUT = httpx.Timeout(connect=30, read=900, write=60, pool=60)   # a single long timeout never fires on a dead socket


def chat(client: httpx.Client, body: dict, retries: int = 4, stage: str = "other") -> dict:
    last = None
    for i in range(retries):
        try:
            r = client.post(CHAT_URL, headers={"Authorization": f"Bearer {api_key()}"}, json=body, timeout=_TIMEOUT)
            if r.status_code in (429, 500, 502, 503, 504):
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            else:
                r.raise_for_status()
                d = r.json()
                cost_ledger.record(COSTS, stage, body.get("model", "?"), d.get("usage"))
                return d
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            last = e
        time.sleep(2 ** i)
    raise RuntimeError(f"chat failed: {last}")


def skill_body() -> str:
    text = (HERE / "sources" / "SKILL.md").read_text()
    if text.startswith("---\n"):
        _, _, text = text.split("---\n", 2)
    return text.strip()


def student_messages(question: str, facts: list[str]) -> list[dict]:
    return [{"role": "system", "content": STUDENT_SYSTEM},
            {"role": "user", "content": question + "\n\nTask information:\n" + "\n".join("- " + f for f in facts)}]


# ============================================================================
# stage 1: tasks
# ============================================================================
THEMES = ["caching", "rate limits", "file systems", "job scheduling", "feature flags", "permissions",
          "data retention", "config precedence", "versioning", "time zones and deadlines", "pagination",
          "retries and backoff", "logging and redaction", "quotas", "units and conversions", "indexes",
          "backups and restores", "queues and ordering", "TLS and certificates", "environment variables",
          "package dependencies", "database migrations", "container limits", "cron and schedules", "webhooks",
          "billing periods and proration", "API deprecations", "locale and formatting", "disk and memory sizing",
          "access tokens and expiry", "shipping and inventory rules", "school and course rules", "library loans",
          "sports league rules", "recipe scaling", "travel itineraries", "parking and permits", "warranty terms"]
THEMES_EVERYDAY = ["cooking and recipe scaling", "travel connections and layovers", "household repairs", "gym and training plans",
                   "personal budgeting", "appointments and scheduling", "gardening", "pet care", "moving house",
                   "event planning", "car maintenance", "study plans", "laundry and fabric care", "medication timing (non-advice, from label facts)"]
AUTHOR_SYSTEM = (
    "You write small, self-contained evaluation fixtures for a technical assistant. Each fixture is a question "
    "plus 2-4 short facts that fully determine the answer. Facts are concrete and invented (no real products, "
    "people or URLs). Return ONLY a JSON object: {\"fixtures\": [{\"group\": <short-kebab-id>, \"question\": ..., "
    "\"facts\": [...], \"required_literals\": [exact tokens any correct answer must contain, e.g. numbers, names, "
    "'not'], \"expected\": <one-line correct answer for checking>}]}."
)


def stage_tasks(n_tasks: int) -> None:
    rng = random.Random(SEED)
    pilot = read_jsonl(HERE / "data" / "candidate_rows.jsonl")
    seen_groups = {r["group_id"] for r in pilot}
    seen_questions = {r["question"].strip().lower() for r in pilot}
    per_call = 8
    jobs = []
    for fam, (share, instr) in FAMILIES.items():
        n = round(n_tasks * share)
        for i in range(0, n, per_call):
            jobs.append((fam, instr, min(per_call, n - i), rng.randint(1, 10**6)))

    def author(job):
        fam, instr, k, seed = job
        themes = THEMES_EVERYDAY if fam == "everyday" else THEMES
        rng2 = random.Random(seed)
        picked = ", ".join(rng2.sample(themes, 4))
        user = (f"Write {k} fixtures of this kind: {instr}. Vary the domain across: {picked}. Seed {seed}. "
                f"Each question must be answerable only from its facts; each 'expected' answer must be correct.")
        with httpx.Client() as c:
            d = chat(c, {"model": AUTHOR_MODEL, "max_tokens": 4096, "temperature": 1.0, "reasoning_effort": "low",
                         "response_format": {"type": "json_object"},
                         "messages": [{"role": "system", "content": AUTHOR_SYSTEM}, {"role": "user", "content": user}]}, stage="author")
        content = d["choices"][0]["message"].get("content") or ""
        try:
            fx = json.loads(re.search(r"\{.*\}", content, re.S).group(0)).get("fixtures", [])
        except (AttributeError, json.JSONDecodeError):
            return []
        out = []
        for f in fx:
            if not isinstance(f, dict) or not all(k in f for k in ("group", "question", "facts", "required_literals", "expected")):
                continue
            facts = f.get("facts")
            if not isinstance(facts, list) or not (2 <= len(facts) <= 4) or not all(isinstance(x, str) and x.strip() for x in facts):
                continue
            if not isinstance(f.get("question"), str) or not f["question"].strip():
                continue
            lits = f.get("required_literals")
            if not isinstance(lits, list):
                continue
            out.append({"family": fam, "group": str(f["group"]), "question": f["question"], "facts": facts,
                        "required_literals": [str(x) for x in lits if str(x).strip()], "expected": str(f["expected"])})
        return out

    with ThreadPoolExecutor(CONCURRENCY) as ex:
        results = [r for rs in ex.map(author, jobs) for r in rs]
    rows, groups = [], {}
    for f in results:
        q = f["question"].strip().lower()
        g = re.sub(r"[^a-z0-9-]", "-", f["group"].lower())[:40] or "fixture"
        if q in seen_questions or g in seen_groups or g in groups:
            g = f"{g}-{len(groups)}"
        seen_questions.add(q)
        groups[g] = True
        rows.append({"id": f"cv-{len(rows) + 1:04d}", "group_id": g, "family": f["family"],
                     "question": f["question"].strip(), "facts": [x.strip() for x in f["facts"]],
                     "required_literals": [str(x) for x in f["required_literals"]], "expected": str(f["expected"]).strip(),
                     "student_messages": student_messages(f["question"].strip(), [x.strip() for x in f["facts"]])})
    # Verify: a different model solves each fixture from its facts alone. Keep a
    # fixture only when that answer agrees with the author's expected answer;
    # an authored fixture can be internally inconsistent, and a wrong expected
    # answer would later make the judge reject correct targets.
    def verify(r):
        user = (f"QUESTION:\n{r['question']}\n\nFACTS:\n" + "\n".join("- " + f for f in r["facts"])
                + f"\n\nSomeone claims the correct answer is: {r['expected']}\n\n"
                "First solve the question yourself using ONLY the facts. Then decide whether the claimed answer is "
                "correct and whether the facts are sufficient and internally consistent. Return ONLY JSON: "
                '{"your_answer": "...", "claim_correct": true|false, "facts_consistent": true|false, "flaw": "..."}')
        with httpx.Client() as c:
            try:
                d = chat(c, {"model": JUDGE_MODEL, "max_tokens": 4096, "temperature": 0.0, "reasoning_effort": "low",
                             "response_format": {"type": "json_object"},
                             "messages": [{"role": "user", "content": user}]}, stage="verify")
                j = json.loads(re.search(r"\{.*\}", d["choices"][0]["message"].get("content") or "", re.S).group(0))
                return bool(j.get("claim_correct")) and bool(j.get("facts_consistent")), str(j.get("flaw", ""))[:200]
            except (RuntimeError, AttributeError, KeyError, ValueError, TypeError, json.JSONDecodeError):
                return False, "verifier failed"
    with ThreadPoolExecutor(CONCURRENCY) as ex:
        verdicts = list(ex.map(verify, rows))
    dropped = [(r["id"], flaw) for r, (ok, flaw) in zip(rows, verdicts) if not ok]
    rows = [r for r, (ok, _) in zip(rows, verdicts) if ok]
    write_jsonl(DATA / "caveman_tasks_dropped.jsonl", [{"id": i, "flaw": f} for i, f in dropped])
    print(f"verification dropped {len(dropped)} fixtures (see data/caveman_tasks_dropped.jsonl)")
    # split by group: 10% dev, 10% test, rest train
    rng.shuffle(rows)
    n_dev = n_test = max(1, len(rows) // 10)
    for i, r in enumerate(rows):
        r["split"] = "test" if i < n_test else ("dev" if i < n_test + n_dev else "train")
    write_jsonl(DATA / "caveman_tasks.jsonl", rows)
    fam = {k: sum(r["family"] == k for r in rows) for k in FAMILIES}
    spl = {k: sum(r["split"] == k for r in rows) for k in ("train", "dev", "test")}
    print(f"wrote {len(rows)} tasks -> {DATA / 'caveman_tasks.jsonl'}  families {fam}  splits {spl}")


# ============================================================================
# stage 2: targets (teacher = base + skill prompt), training fixtures only
# ============================================================================
def stage_targets() -> None:
    tasks = [t for t in require_tasks() if t["split"] == "train"]
    out = DATA / "caveman_targets.jsonl"
    fps = {t["id"]: input_fp(t) for t in tasks}
    tfps = {t["id"]: teacher_fp(t["family"]) for t in tasks}
    done = {(r["task_id"], r["sample_index"]) for r in load_cache(out, fps, "task_id", "teacher_fp", tfps)}
    suffix = "\n\n" + skill_body() + SKILL_SUFFIX
    jobs = [(t, i) for t in tasks for i in range(TARGETS_PER_TASK) if (t["id"], i) not in done]
    # Family hint for the teacher. The skill-prompted base answers an unanswerable question correctly but then
    # appends "check the docs" and illustrative numbers; the judge's on_request axis and the number check reject
    # most of those, so the family starves (92 of 1,312 targets in the first build). Say the rule once.
    family_hint = {"cannot_tell": ("\n\nIf the task information cannot answer the question: say so, name exactly what is "
                                   "missing, and stop. No suggestions where to look, no example values, no procedure.")}
    print(f"{len(tasks)} training tasks x {TARGETS_PER_TASK} teacher samples; {len(jobs)} to generate")

    def gen(job):
        t, i = job
        msgs = [dict(m) for m in t["student_messages"]]
        msgs[0]["content"] += suffix + family_hint.get(t["family"], "")
        with httpx.Client() as c:
            d = chat(c, {"model": BASE_MODEL, "max_tokens": TEACHER_MAX_TOKENS, "temperature": TEACHER_TEMPERATURE,
                         "reasoning_effort": TEACHER_EFFORT, "messages": msgs}, stage="teacher")
        ch = d["choices"][0]
        return {"task_id": t["id"], "sample_index": i, "input_fp": fps[t["id"]], "teacher_fp": tfps[t["id"]],
                "text": (ch["message"].get("content") or "").strip(),
                "finish": ch.get("finish_reason"), "completion_tokens": (d.get("usage") or {}).get("completion_tokens")}

    t0 = time.time()
    with out.open("a") as f, ThreadPoolExecutor(CONCURRENCY) as ex:
        for n, rec in enumerate((fu.result() for fu in as_completed([ex.submit(gen, j) for j in jobs])), 1):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            if n % 50 == 0 or n == len(jobs):
                print(f"  [{n}/{len(jobs)}] +{time.time() - t0:.0f}s", flush=True)
    print(f"{len(read_jsonl(out))} teacher samples -> {out}")


# ============================================================================
# stage 3: filter
# ============================================================================
_NUM = re.compile(r"\d+(?:[.,:]\d+)*")


def _norm(s: str) -> str:
    """Lowercase, drop thousands separators and surrounding quotes/backticks, collapse whitespace."""
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s.lower())
    return re.sub(r"\s+", " ", s.replace("`", "").replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")).strip()


REFUSAL_WORDS = {"cannot", "can't", "not", "no", "never", "determined", "determine", "specified", "unspecified",
                 "unknown", "insufficient", "missing", "unclear", "undetermined", "not specified", "not enough",
                 "cannot be determined", "not given", "not stated", "unable", "impossible", "none"}


def effective_literals(task: dict) -> list[str]:
    """The literals an answer must contain. For a cannot_tell fixture the author tends to require its own refusal
    wording ("cannot", "determined", "not specified"); a correct refusal says the same thing in other words, and
    the judge's `complete` axis already checks that the answer declines and names what is missing. So for that
    family only the non-refusal literals (the names of the missing things) are enforced in code."""
    lits = task["required_literals"]
    if task.get("family") == "cannot_tell":
        return [l for l in lits if _norm(l) not in REFUSAL_WORDS]
    return lits


def literal_checks(text: str, task: dict) -> list[str]:
    """Code-only checks from the pilot review.

    Required literals must be present (ignoring case, backticks and thousands
    separators). A number may appear only if it is in the input OR in the
    fixture's own expected answer: that admits correct derived arithmetic
    ("80 / 10 = 8") while still rejecting invented figures.
    """
    problems = []
    low = _norm(text)
    negations = ("not", "no", "never", "n't", "none", "denied", "deny", "cannot", "can't")
    required = effective_literals(task)
    other_lits = [_norm(l) for l in required if _norm(l) not in ("not", "no", "never")]
    for lit in required:
        n = _norm(lit)
        if n in ("not", "no", "never"):
            # A negation requirement is satisfied by a negation word, but only within a short window
            # (three words) of another required literal when there is one, so "Yes. No limit applies."
            # does not satisfy a fixture whose answer is "not supported".
            neg = r"\b(?:" + "|".join(re.escape(w) for w in negations) + r")\b"
            # Polarity by position: the answer's first word is a negation ("No. ...", "Not supported."),
            # or a negation sits within three words of another required literal ("is not supported",
            # "supported? no"). A reply that opens with "yes" fails a negation requirement outright.
            # A literal check cannot prove polarity in general; the judge's `complete` axis owns the rest.
            leading = re.match(r"\W*" + neg, low) is not None
            affirms = re.match(r"\W*(?:yes|correct|true|right)\b", low) is not None
            near = r"(?:\W+\w+){0,3}\W+"
            if affirms:
                ok = False
            elif leading or not other_lits:
                ok = leading or re.search(neg, low) is not None
            else:
                ok = any(re.search(neg + near + re.escape(o), low) or re.search(re.escape(o) + near + neg, low) for o in other_lits)
            if not ok:
                problems.append(f"missing negation (literal {lit})")
        elif n not in low:
            problems.append(f"missing literal: {lit}")
    # Numbers are compared as whole tokens: "8" is not allowed just because "80s" is in the facts. A
    # dotted or dashed component of an allowed token is fine ("10.1" from "10.1.5", "02" from "2024-02-29"),
    # because version and date reasoning names the parts.
    allowed_text = _norm(task["question"] + " " + " ".join(task["facts"]) + " " + task["expected"])
    allowed_nums = set(_NUM.findall(allowed_text))
    components = {c for tok in re.findall(r"[\d.\-:]+", allowed_text) for c in re.split(r"[.\-:]", tok) if c}
    components |= {".".join(parts[:k]) for tok in allowed_nums for parts in [tok.split(".")] for k in range(1, len(parts))}
    for n in set(_NUM.findall(low)):
        if n not in allowed_nums and n not in components:
            problems.append(f"number not in input or expected answer: {n}")
    if not text.strip() or len(text.split()) > 220:
        problems.append("empty or over 220 words")
    return problems


JUDGE_SYS = "You check a terse assistant answer against the task it was given. Return ONLY a JSON object."


def judge(client: httpx.Client, text: str, task: dict) -> dict | None:
    user = (f"QUESTION:\n{task['question']}\n\nFACTS (the only allowed information):\n" + "\n".join("- " + f for f in task["facts"])
            + f"\n\nEXPECTED ANSWER (for your reference): {task['expected']}\n\nANSWER TO CHECK:\n{text}\n\n"
            "Rate each 0.0-1.0:\n- supported: every claim in the answer follows from the facts; deduct for anything invented.\n"
            "- complete: the answer actually answers the question with the facts needed; deduct for a missing fact or a wrong conclusion.\n"
            "- on_request: the answer does only what was asked (no extra advice, procedures or sections) and honors any explicit format or "
            "style requirement in the question exactly (e.g. 'full sentences', 'return only the CSV', 'no pronouns').\n"
            "Terse, article-dropping style is expected and is NOT a defect. Think step by step, then output ONLY the JSON.\n"
            'Format (illustrative, do not copy): {"supported":0.9,"complete":1.0,"on_request":0.7}')
    try:
        d = chat(client, {"model": JUDGE_MODEL, "max_tokens": 4096, "temperature": 0.0, "reasoning_effort": "low",
                          "response_format": {"type": "json_object"},
                          "messages": [{"role": "system", "content": JUDGE_SYS}, {"role": "user", "content": user}]}, stage="judge")
        c = d["choices"][0]["message"].get("content") or ""
        j = json.loads(re.search(r"\{.*\}", c, re.S).group(0))
        return {k: max(0.0, min(1.0, float(j[k]))) for k in ("supported", "complete", "on_request")}
    except (RuntimeError, AttributeError, KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


def stage_filter() -> None:
    tasks = {t["id"]: t for t in require_tasks()}
    fps = {i: input_fp(t) for i, t in tasks.items()}
    samples = load_cache(DATA / "caveman_targets.jsonl", fps, "task_id", "teacher_fp", {i: teacher_fp(t["family"]) for i, t in tasks.items()})
    out = DATA / "caveman_scores.jsonl"
    jfp = judge_fp()
    prev = load_cache(out, fps, "task_id", "judge_fp", jfp, drop_status="judge_failed")
    # A score is bound to the exact target text it judged. If the teacher config changed and a
    # slot now holds a new answer, the old score is dropped and the new text is judged.
    current = {(s["task_id"], s["sample_index"]): s["text"] for s in samples}
    stale = [r for r in prev if current.get((r["task_id"], r["sample_index"])) != r["text"]]
    if stale:
        prev = [r for r in prev if r not in stale]
        print(f"  cache {out.name}: dropped {len(stale)} rows whose target text changed")
        write_jsonl(out, prev)
    done = {(r["task_id"], r["sample_index"]) for r in prev}
    jobs = [s for s in samples if (s["task_id"], s["sample_index"]) not in done]
    print(f"{len(jobs)} samples to check ({len(done)} done)")

    def check(s):
        t = tasks[s["task_id"]]
        rec = {"task_id": s["task_id"], "sample_index": s["sample_index"], "input_fp": s["input_fp"], "judge_fp": jfp,
               "text": s["text"], "words": len(s["text"].split())}
        probs = literal_checks(s["text"], t)
        if s.get("finish") != "stop":
            probs.append("truncated")
        if probs:
            rec.update(status="rejected_by_code", problems=probs, judge=None)
            return rec
        with httpx.Client() as c:
            j = judge(c, s["text"], t)
        if j is None:
            rec.update(status="judge_failed", problems=[], judge=None)
            return rec
        bad = [k for k, v in j.items() if v < JUDGE_MIN]
        rec.update(status="accepted" if not bad else "rejected_by_judge", problems=[f"{k}={j[k]:.1f}" for k in bad], judge=j)
        return rec

    t0 = time.time()
    with out.open("a") as f, ThreadPoolExecutor(CONCURRENCY) as ex:
        for n, rec in enumerate((fu.result() for fu in as_completed([ex.submit(check, j) for j in jobs])), 1):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            if n % 50 == 0 or n == len(jobs):
                print(f"  [{n}/{len(jobs)}] +{time.time() - t0:.0f}s", flush=True)
    rows = read_jsonl(out)
    st = {}
    for r in rows:
        st[r["status"]] = st.get(r["status"], 0) + 1
    print(f"{len(rows)} checked -> {out}  {st}")


# ============================================================================
# stage 4: pack
# ============================================================================
def stage_pack() -> None:
    tasks = require_tasks()
    scores = load_cache(DATA / "caveman_scores.jsonl", {t["id"]: input_fp(t) for t in tasks}, "task_id", "judge_fp", judge_fp())
    best: dict[str, dict] = {}
    for r in scores:
        if r["status"] != "accepted":
            continue
        cur = best.get(r["task_id"])
        if cur is None or r["words"] < cur["words"]:
            best[r["task_id"]] = r
    train = []
    for t in tasks:
        if t["split"] != "train" or t["id"] not in best:
            continue
        train.append({"messages": t["student_messages"] + [{"role": "assistant", "content": best[t["id"]]["text"]}],
                      "meta": {"task_id": t["id"], "group_id": t["group_id"], "family": t["family"]}})
    evals = {s: [{"id": t["id"], "group_id": t["group_id"], "family": t["family"], "question": t["question"], "facts": t["facts"],
                  "required_literals": t["required_literals"], "expected": t["expected"], "student_messages": t["student_messages"]}
                 for t in tasks if t["split"] == s] for s in ("dev", "test")}
    if not train:
        raise SystemExit("no accepted training targets: run `targets` and `filter` first; the frozen pack is left untouched")
    PACK.mkdir(exist_ok=True)
    write_jsonl(PACK / "caveman_train.jsonl", train)
    write_jsonl(PACK / "caveman_dev.jsonl", evals["dev"])
    write_jsonl(PACK / "caveman_test.jsonl", evals["test"])
    n_train_tasks = sum(t["split"] == "train" for t in tasks)
    st = {}
    for r in scores:
        st[r["status"]] = st.get(r["status"], 0) + 1
    manifest = {
        "method": "self-distillation: teacher = base model + Caveman skill prompt; student prompt has no skill",
        "base_model": BASE_MODEL, "author_model": AUTHOR_MODEL, "judge_model": JUDGE_MODEL,
        "teacher": {"temperature": TEACHER_TEMPERATURE, "samples_per_task": TARGETS_PER_TASK, "reasoning_effort": TEACHER_EFFORT, "max_tokens": TEACHER_MAX_TOKENS},
        "costs": cost_ledger.summary(COSTS),
        "fixture_verification": "a second model (judge_model) solved each fixture from its facts; fixtures whose expected answer or facts it disputed were dropped",
        "filter": {"code": "required literals present (case/thousands-separator insensitive); every number appears in question+facts or in the expected answer; <= 220 words; finish=stop",
                   "judge": f"supported, complete, on_request each >= {JUDGE_MIN}", "pick": "shortest accepted sample"},
        "tasks": {"total": len(tasks), "train": n_train_tasks, "dev": len(evals["dev"]), "test": len(evals["test"]),
                  "families": {k: sum(t["family"] == k for t in tasks) for k in FAMILIES}},
        "train_rows": len(train), "train_yield": round(len(train) / max(1, n_train_tasks), 3), "sample_statuses": st,
        "train_sha256": sha256(PACK / "caveman_train.jsonl"), "test_sha256": sha256(PACK / "caveman_test.jsonl"),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (PACK / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"train {len(train)} rows from {n_train_tasks} tasks (yield {manifest['train_yield']:.0%}); "
          f"dev {len(evals['dev'])}, test {len(evals['test'])} -> {PACK}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["tasks", "targets", "filter", "pack", "all"])
    ap.add_argument("--n-tasks", type=int, default=N_TASKS)
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    for s in (["tasks", "targets", "filter", "pack"] if a.stage == "all" else [a.stage]):
        print(f"\n===== {s} =====")
        {"tasks": lambda: stage_tasks(a.n_tasks), "targets": stage_targets, "filter": stage_filter, "pack": stage_pack}[s]()
        print("costs so far:"); cost_ledger.print_summary(COSTS)


if __name__ == "__main__":
    main()
