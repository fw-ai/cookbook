"""Score all visible text: useful English answers with no e, escapes or look-alikes.

Strict is the default. Ramp is an experimental cold-start comparison, using
absolute violation count so appending clean filler cannot improve the reward.
Quality is judged separately; short answers are valid when the task is answered.
"""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata

import httpx

CHAT_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
JUDGE_MODEL = "accounts/fireworks/models/deepseek-v4p1-flash"
JUDGE_EFFORT = "high"
JUDGE_MAX_TOKENS = 8192
ANSWER_MIN = 0.5      # judge's completeness score below which a reply earns nothing
PARTIAL = 0.4         # ceiling, as a share of the completeness score, for a reply that still contains e's

JUDGE_SYS = (
    "You grade whether a reply answers the user's request. Ignore style, spelling oddities and unusual "
    "word choices entirely: the writer is working under a constraint and may phrase things strangely. "
    "Score only substance. A longer reply that covers more of what the reader needs scores higher than "
    "a shorter one that covers less; never reward brevity for its own sake. Output JSON only."
)


# Anything that reads as an "e" counts as one: Latin e/E, every accented form (after NFKD these
# become e + combining mark, but list them anyway), Cyrillic е/Е/ё, the estimated sign, fullwidth
# forms, schwa. The first trained policy found "thе" with a Cyrillic е and "\u0065" escapes.
_E_LOOKALIKE = re.compile(
    r"[eE\u0435\u0415\u0450\u0451\u00e8-\u00eb\u00c8-\u00cb\u0113-\u011b\u0112-\u011a\u1eb8-\u1ec7"
    r"\u212e\u212f\u2147\uff45\uff25\u0259\u018f]"
)
_ESCAPED_E = re.compile(r"\\u00(?:65|45)|&#(?:101|69|x65|x45);", re.IGNORECASE)
# Letters outside Latin script. Switching to Hungarian, Chinese or Cyrillic to dodge the letter is a
# slip, not a solution: the user asked in English.
_NON_LATIN_LETTER = re.compile(r"[^\W\d_]")


_INVISIBLE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]")   # zero-width and soft-hyphen characters


def _canonical(text: str) -> str:
    """NFKD-normalize, drop invisible characters, and turn any escaped or look-alike e into a plain e."""
    text = _INVISIBLE.sub("", text)
    text = _ESCAPED_E.sub("e", text)
    text = unicodedata.normalize("NFKD", text)
    return _E_LOOKALIKE.sub("e", text)


def non_latin_letters(text: str) -> int:
    """Count letters that are not Latin script (after stripping code blocks)."""
    n = 0
    for ch in _NON_LATIN_LETTER.findall(text):
        try:
            if not unicodedata.name(ch).startswith("LATIN"):
                n += 1
        except ValueError:
            n += 1
    return n


def count_e(text: str) -> tuple[int, int]:
    """(number of e's including every look-alike, number of letters) in ``text``."""
    t = _canonical(text)
    letters = sum(1 for ch in t if ch.isalpha())
    return t.count("e") + t.count("E") + non_latin_letters(t), letters


def e_free(text: str) -> bool:
    t = _canonical(text)
    return "e" not in t and "E" not in t and non_latin_letters(t) == 0


def e_rate_per_100(text: str) -> float:
    es, letters = count_e(text)
    return 100.0 * es / letters if letters else 0.0


def judge_user(question: str, answer: str, facts: str = "") -> str:
    ctx = f"\nContext the answer may rely on:\n{facts}\n" if facts else ""
    return (
        f"REQUEST:\n{question}\n{ctx}\nREPLY:\n{answer}\n\n"
        "Rate 0.0-1.0:\n"
        "- answers: the reply addresses the request and gives the substance a reader needs, with the "
        "detail a careful answer would include. A reply that dodges, refuses, or says almost nothing scores "
        "near 0; a partial answer scores in the middle; a thorough one scores near 1. Strange phrasing is fine.\n"
        "- faithful: factual correctness and consistency with supplied context. Without context, "
        "still check factual claims; do not automatically assign 1.0.\n"
        'Output ONLY: {"answers": 0.9, "faithful": 1.0}'
    )


def chat(client: httpx.Client, body: dict, retries: int = 4) -> dict:
    last: Exception | None = None
    for i in range(retries):
        try:
            r = client.post(CHAT_URL, headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"},
                            json=body, timeout=httpx.Timeout(connect=30.0, read=900.0, write=60.0, pool=60.0))
            if r.status_code in (429, 500, 502, 503, 504):
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            else:
                r.raise_for_status()
                return r.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            last = e
        time.sleep(2 ** i)
    raise RuntimeError(f"chat failed after {retries} tries: {last}")


def _judge_once(client: httpx.Client, question: str, answer: str, facts: str = "") -> dict | None:
    body = {"model": JUDGE_MODEL, "max_tokens": JUDGE_MAX_TOKENS, "temperature": 0.0,
            "reasoning_effort": JUDGE_EFFORT, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": JUDGE_SYS},
                         {"role": "user", "content": judge_user(question, answer, facts)}]}
    try:
        d = chat(client, body)
        c = d["choices"][0]["message"].get("content") or ""
        m = re.search(r"\{.*\}", c, re.S)
        j = json.loads(m.group(0)) if m else {}
        return {k: max(0.0, min(1.0, float(j[k]))) for k in ("answers", "faithful")}
    except (RuntimeError, KeyError, ValueError, TypeError):
        return None


JUDGE_CALLS = 2   # independent calls averaged; the judge decides what enters the training set


def _judge_once(client: httpx.Client, question: str, answer: str, facts: str = "") -> dict | None:
    body = {"model": JUDGE_MODEL, "max_tokens": JUDGE_MAX_TOKENS, "temperature": 0.0,
            "reasoning_effort": JUDGE_EFFORT, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": JUDGE_SYS},
                         {"role": "user", "content": judge_user(question, answer, facts)}]}
    try:
        d = chat(client, body)
        c = d["choices"][0]["message"].get("content") or ""
        m = re.search(r"\{.*\}", c, re.S)
        j = json.loads(m.group(0)) if m else {}
        return {k: max(0.0, min(1.0, float(j[k]))) for k in ("answers", "faithful")}
    except (RuntimeError, KeyError, ValueError, TypeError):
        return None


JUDGE_CALLS = 2   # independent calls averaged; the judge decides what enters the training set


def llm_judge(client: httpx.Client, question: str, answer: str, facts: str = "") -> dict | None:
    """Mean of JUDGE_CALLS verdicts ({"answers": x, "faithful": y} in [0,1]); None if any call failed."""
    verdicts = [_judge_once(client, question, answer, facts) for _ in range(JUDGE_CALLS)]
    if any(v is None for v in verdicts):
        return None
    return {k: sum(v[k] for v in verdicts) / len(verdicts) for k in ("answers", "faithful")}


def lipogram_reward(answer: str, judge: dict | None, phase: str = "strict") -> tuple[float, bool, bool, float]:
    """(reward, e_free, answered, e_rate) given the visible answer, a judge verdict and the phase.

    ``judge`` None means the scorer failed; the caller drops the group (see antislop_rl).
    ``phase`` is "ramp" (partial credit for slips) or "strict" (slips earn nothing).
    """
    if phase not in ("ramp", "strict"):
        raise ValueError(f"Unknown reward phase: {phase}")
    rate = e_rate_per_100(answer)
    if not answer.strip():
        return 0.0, False, False, rate
    free = e_free(answer)
    if judge is None or judge["answers"] < ANSWER_MIN or judge["faithful"] < ANSWER_MIN:
        return 0.0, free, False, rate
    score = judge["answers"]
    if free:
        return score, True, True, rate
    if phase == "strict":
        return 0.0, False, True, rate
    return score * PARTIAL / (1 + count_e(answer)[0]), False, True, rate
