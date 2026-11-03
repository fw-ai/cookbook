#!/usr/bin/env python3
"""Answer normally, then rewrite e-free: the lipogram SFT data pipeline.

Per seed (seeds.py):
  1. draft    GLM 5.3 Flash, effort low: restyle the seed for its persona / type / length and answer it
              normally (JSON {"question", "answer"}).
  2. rewrite  the rewriter (Kimi K3 or GLM at max effort) rewrites the answer with no letter e, given
              e-free WordNet synonyms for the e-words it contains.
  3. fix      Rob's strict checker (look-alike, escaped, invisible and non-Latin letters count) lists the
              words that still break the rule; the rewriter repairs them, up to --fix-rounds times.
              The repair conversation exists only here: the exported row is question -> final answer.
  4. judge    (a) fidelity to the original answer and naturalness (this script's judge), and
              (b) Rob's llm_judge: does it answer the request, is it factually right.
  5. accept   e-free, no non-Latin letters, >= half the original's words, every e-free name and digit
              number of the original kept (checks.missing_keepables), no talk about the constraint
              (checks.talks_about_constraint and the judge), the judge lists no material error and no
              unrecognizable name, fidelity >= 0.8, natural >= 0.65, answers >= 0.7, faithful >= 0.7.

One JSON line per seed goes to <out>/attempts.jsonl (every stage, token usage, cost, verdicts);
seeds already present are skipped, so a run resumes. Running this makes paid API calls.

    python pipeline.py --seeds data/seeds.jsonl --rewriter kimi --limit 40 --out runs/cmp-kimi
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rl_lipogram"))
import lipogram_rewards  # noqa: E402
from lipogram_rewards import _canonical, e_free, llm_judge, non_latin_letters  # noqa: E402

from checks import missing_keepables, talks_about_constraint  # noqa: E402
from lexicon import hints_for  # noqa: E402

URL = "https://api.fireworks.ai/inference/v1/chat/completions"
GLM = "accounts/fireworks/models/glm-5p3-flash"
KIMI = "accounts/fireworks/models/kimi-k3"
JUDGE = "accounts/fireworks/models/deepseek-v4p1-flash"
PRICES = {GLM: (0.15, 0.50), KIMI: (3.00, 15.00), JUDGE: (0.30, 1.20)}   # $/1M input, output

DRAFT_SYS = "You write realistic user requests and give genuinely helpful, accurate answers."
DRAFT_USER = """Seed request (from a real user):
<<<
{text}
>>>

1. Rewrite the seed as a natural message from {persona}, as a {type} request, that implicitly or explicitly wants
   an answer of about {length_text}. Keep the topic and any supplied passage; it must read like a real chat message.
2. Answer that message as a helpful assistant would, at that length. Plain prose or simple lists; no headings.

Return JSON only: {{"question": "...", "answer": "..."}}"""

REWRITE_SYS = """You are an expert at lipograms. Rewrite the given answer so that the letter "e" (lowercase or capital)
appears NOWHERE in your output: not in any word, heading, list marker, name or number word. Use digits for numbers.
Keep every fact, step and caveat of the original, keep roughly its length and structure, and write natural, fluent
English a reader can follow. Do not use look-alike characters, escapes, accents, other alphabets or invented words.
Names and numbers: copy every name that has no "e" exactly as written, and keep every number (as digits). For a name
that contains an "e", use a form a reader still recognizes unambiguously (initials, a well-known short form, or a clear
description such as "Isaac N., who wrote down gravity's laws"); never swap in a different or made-up name.
Never mention the letter, the rule, a constraint, or that you are avoiding or skipping anything. Just answer.
Return only the rewritten answer, nothing else."""

REWRITE_USER = """Question the answer responds to:
<<<
{question}
>>>

Original answer:
<<<
{answer}
>>>

Possible e-free replacements for some words (use or ignore):
{hints}"""

FIX_USER = """These words in your text still contain the letter e: {words}.
Rewrite only what is needed so that the whole text contains no e at all, keeping the meaning and fluency.
Return the full corrected text only."""

RESTORE_USER = """Your text dropped or changed these names or numbers from the original: {items}.
Put each one back exactly as written (they contain no e), keep everything else, and keep the whole text free of the
letter e. Return the full corrected text only."""

REPAIR_USER = """A careful reader found these problems in your text compared with the original:
{notes}
Fix each one so the text says what the original says (use digits, initials or a clear description where a word with
an e is unavoidable), keep the rest, and keep the whole text free of the letter e. Never mention the rule.
Return the full corrected text only."""

FIDELITY_SYS = "You compare a rewritten answer with the original it was rewritten from. Output JSON only."
FIDELITY_USER = """QUESTION:
{question}

ORIGINAL ANSWER:
{original}

REWRITTEN ANSWER (written under a constraint: no letter e):
{rewrite}

The rewrite had to avoid one letter, so some words are necessarily swapped for near-synonyms. Judge it the way a
careful reader would: would they come away with a wrong idea? Compare sentence by sentence, then output JSON:
- "material_errors": short notes, one per place where the rewrite would MISLEAD a reader: a wrong term or concept
  (e.g. a different device, cell type, language, material or process), a changed number, date or amount, a lost or
  flipped condition or contrast, or an added claim that is false or unsupported. Do NOT list near-synonyms, slightly
  less precise wording, or rephrasings that keep the meaning. Empty list if none.
- "minor_imprecisions": an integer count of near-synonym swaps that are slightly less precise but not misleading.
- "unrecognizable_names": names (people, places, works, organisations, products, languages) from the original that
  the rewrite replaces with a different or made-up name, or disguises so that a reader could not work out which one is
  meant. A recognizable short form, initials with context, or an unambiguous description is fine. Empty if none.
- "mentions_constraint": true if the rewrite mentions or alludes to avoiding a letter, symbol or character, or to any
  writing rule or limitation; otherwise false.
- "fidelity": 0.0-1.0, how completely and correctly the rewrite keeps the original's content.
- "natural": 0.0-1.0, how fluent and easy to read it is. Slightly unusual vocabulary is fine; broken grammar, garbled
  or invented words, odd substitutes that obscure meaning, or telegraphic fragments score low.
Output ONLY the JSON object."""


class Ledger:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.by_model: dict[str, list[float]] = {}

    def add(self, model: str, usage: dict) -> float:
        p_in, p_out = PRICES.get(model, PRICES[GLM])
        c = (usage.get("prompt_tokens", 0) * p_in + usage.get("completion_tokens", 0) * p_out) / 1e6
        with self.lock:
            v = self.by_model.setdefault(model, [0.0, 0, 0])
            v[0] += c
            v[1] += usage.get("prompt_tokens", 0)
            v[2] += usage.get("completion_tokens", 0)
        return c


LEDGER = Ledger()
_LOCAL = threading.local()


def _charge(model: str, usage: dict) -> None:
    """Add one call's cost to the global ledger and to the current seed's own total."""
    _LOCAL.cost = getattr(_LOCAL, "cost", 0.0) + LEDGER.add(model, usage)


_rob_chat = lipogram_rewards.chat


def _rob_chat_charged(client, body, retries=4):
    d = _rob_chat(client, body, retries)
    _charge(body["model"], d.get("usage") or {})
    return d


lipogram_rewards.chat = _rob_chat_charged   # Rob's llm_judge looks chat up at call time


def chat(client: httpx.Client, body: dict, tries: int = 8) -> tuple[str, dict, str]:
    """(content, usage, finish_reason); streams so long generations survive the gateway."""
    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    err = ""
    for i in range(tries):
        try:
            content, usage, finish = [], {}, None
            with client.stream("POST", URL, headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"},
                               json=body, timeout=httpx.Timeout(connect=30, read=900, write=60, pool=60)) as r:
                if r.status_code != 200:
                    err = f"HTTP {r.status_code}: {r.read()[:200]!r}"
                    if r.status_code in (400, 401, 403):
                        raise RuntimeError(err)
                    time.sleep(min(90, 5 * 2 ** i))
                    continue
                for raw in r.iter_bytes():
                    content.append(raw)
            text = b"".join(content).decode("utf-8")
            out = []
            for line in text.split("\n"):
                line = line.strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                ch = json.loads(line[5:])
                if ch.get("usage"):
                    usage = ch["usage"]
                for c in ch.get("choices") or []:
                    out.append((c.get("delta") or {}).get("content") or "")
                    finish = c.get("finish_reason") or finish
            _charge(body["model"], usage)
            return "".join(out), usage, finish or ""
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001  dropped connections under load surface as several types
            err = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(min(90, 5 * 2 ** i))
    raise RuntimeError(f"chat failed: {err}")


def offending_words(text: str) -> list[str]:
    words = re.findall(r"\S+", _canonical(text))
    bad = [w.strip(".,;:!?()[]{}\"'*") for w in words if "e" in w or "E" in w or non_latin_letters(w)]
    return list(dict.fromkeys(b for b in bad if b))


def json_obj(s: str) -> dict:
    m = re.search(r"\{.*\}", s, re.S)
    return json.loads(m.group(0)) if m else {}


def rewriter_body(model: str, messages: list[dict]) -> dict:
    if model == GLM:
        return {"model": GLM, "messages": messages, "max_tokens": 32768, "temperature": 0.7, "reasoning_effort": "max"}
    return {"model": model, "messages": messages, "max_tokens": 16384, "temperature": 0.7}


def judge_all(client: httpx.Client, question: str, original: str, text: str) -> dict:
    """Both judges on one rewrite; returns the verdict fields, reject_reasons, or a fail key."""
    out: dict = {}
    fc, _, _ = chat(client, {"model": JUDGE, "max_tokens": 16384, "temperature": 0.0, "reasoning_effort": "high",
                             "response_format": {"type": "json_object"},
                             "messages": [{"role": "system", "content": FIDELITY_SYS},
                                          {"role": "user", "content": FIDELITY_USER.format(
                                              question=question, original=original, rewrite=text)}]})
    try:
        fj = json_obj(fc)
        out["fidelity"], out["natural"] = float(fj["fidelity"]), float(fj["natural"])
        out["material_errors"] = [x for x in (fj.get("material_errors") or []) if str(x).strip()]
        out["minor_imprecisions"] = int(fj.get("minor_imprecisions") or 0)
        out["unrecognizable_names"] = [x for x in (fj.get("unrecognizable_names") or []) if str(x).strip()]
        out["mentions_constraint"] = bool(fj.get("mentions_constraint"))
    except (KeyError, ValueError, TypeError):
        return {"fail": "fidelity_judge_failed"}
    rob = llm_judge(client, question, text)
    if rob is None:
        return {"fail": "rob_judge_failed"}
    out["answers"], out["faithful"] = rob["answers"], rob["faithful"]
    reasons = [k for k, bad in (("material_errors", out["material_errors"]),
                                ("unrecognizable_names", out["unrecognizable_names"]),
                                ("mentions_constraint", out["mentions_constraint"]),
                                ("fidelity", out["fidelity"] < 0.8), ("natural", out["natural"] < 0.65),
                                ("answers", out["answers"] < 0.7), ("faithful", out["faithful"] < 0.7)) if bad]
    if reasons:
        out["reject_reasons"] = reasons
    return out


def _process(seed: dict, rewriter: str, fix_rounds: int) -> dict:
    rec: dict = {"id": seed["id"], "source": seed["source"], "type": seed["type"], "length": seed["length"],
                 "persona": seed["persona"], "rewriter": rewriter, "stage": "draft"}
    _LOCAL.cost = 0.0
    t0 = time.time()
    with httpx.Client() as client:
        c, _, _ = chat(client, {"model": GLM, "max_tokens": 4096, "temperature": 0.8, "reasoning_effort": "low",
                                "response_format": {"type": "json_object"},
                                "messages": [{"role": "system", "content": DRAFT_SYS},
                                             {"role": "user", "content": DRAFT_USER.format(**seed)}]})
        d = json_obj(c)
        question, original = (d.get("question") or "").strip(), (d.get("answer") or "").strip()
        rec.update(question=question, original=original)
        if not question or not original:
            rec["fail"] = "draft_unparsed"
            return rec
        rec["stage"] = "rewrite"
        hints = hints_for(original)
        msgs = [{"role": "system", "content": REWRITE_SYS},
                {"role": "user", "content": REWRITE_USER.format(question=question, answer=original, hints=hints or "(none)")}]
        text, _, finish = chat(client, rewriter_body(rewriter, msgs))
        text = text.strip()
        history = [{"round": 0, "violations": len(offending_words(text)), "finish": finish}]
        for rnd in range(1, fix_rounds + 1):
            bad = offending_words(text)
            if not bad and text:
                break
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": FIX_USER.format(words=", ".join(bad[:60]))}]
            text, _, finish = chat(client, rewriter_body(rewriter, msgs))
            text = text.strip()
            history.append({"round": rnd, "violations": len(offending_words(text)), "finish": finish})
        rec.update(rewrite=text, fix_history=history, e_free=bool(text) and e_free(text),
                   words_original=len(original.split()), words_rewrite=len(text.split()))
        if not rec["e_free"]:
            rec["fail"] = "not_e_free"
            rec["stage"] = "checker"
            return rec
        if rec["words_rewrite"] < 0.5 * rec["words_original"]:
            rec["fail"] = "too_short"
            rec["stage"] = "length"
            return rec
        missing = missing_keepables(original, text)
        if missing:
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": RESTORE_USER.format(items=", ".join(missing[:40]))}]
            fixed, _, finish = chat(client, rewriter_body(rewriter, msgs))
            fixed = fixed.strip()
            history.append({"round": "restore", "missing": len(missing), "finish": finish})
            if fixed and e_free(fixed) and len(fixed.split()) >= 0.5 * rec["words_original"]:
                text = fixed
                rec.update(rewrite=text, words_rewrite=len(text.split()))
        rec["missing_keepables"] = missing_keepables(original, text)
        if rec["missing_keepables"]:
            rec["fail"] = "name_or_number_dropped"
            rec["stage"] = "names"
            return rec
        if talks_about_constraint(text):
            rec["fail"] = "talks_about_constraint"
            rec["stage"] = "meta"
            return rec
        rec["stage"] = "judge"
        verdict = judge_all(client, question, original, text)
        if "fail" in verdict:
            rec.update(verdict)
            return rec
        rec.update(verdict)
        fixable = set(rec.get("reject_reasons", [])) - {"fidelity", "natural"}
        if rec.get("reject_reasons") and fixable <= {"material_errors", "unrecognizable_names"} and fixable:
            notes = [*rec["material_errors"], *[f"name not recognizable: {n}" for n in rec["unrecognizable_names"]]]
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": REPAIR_USER.format(notes="\n".join(f"- {x}" for x in notes[:12]))}]
            fixed, _, finish = chat(client, rewriter_body(rewriter, msgs))
            fixed = fixed.strip()
            history.append({"round": "repair", "notes": len(notes), "finish": finish})
            if (fixed and e_free(fixed) and len(fixed.split()) >= 0.5 * rec["words_original"]
                    and not missing_keepables(original, fixed) and not talks_about_constraint(fixed)):
                rec["first_verdict"] = {k: rec[k] for k in ("material_errors", "unrecognizable_names", "fidelity",
                                                            "natural", "reject_reasons")}
                text = fixed
                rec.update(rewrite=text, words_rewrite=len(text.split()))
                rec.pop("reject_reasons", None)
                verdict = judge_all(client, question, original, text)
                if "fail" in verdict:
                    rec.update(verdict)
                    return rec
                rec.update(verdict)
                rec["repaired"] = True
        rec["accepted"] = not rec.get("reject_reasons")
        rec["stage"] = "done"
        if not rec["accepted"]:
            rec["fail"] = "judge_reject"
        else:
            rec.pop("fail", None)
    return rec


def process(seed: dict, rewriter: str, fix_rounds: int) -> dict:
    t0 = time.time()
    rec = _process(seed, rewriter, fix_rounds)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["cost"] = round(_LOCAL.cost, 5)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default=str(HERE / "data" / "seeds.jsonl"))
    ap.add_argument("--rewriter", choices=["kimi", "glm"], default="kimi")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fix-rounds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rewriter = KIMI if a.rewriter == "kimi" else GLM
    seeds = [json.loads(l) for l in Path(a.seeds).read_text().splitlines()][a.offset:]
    if a.limit:
        seeds = seeds[:a.limit]
    out = Path(a.out) if Path(a.out).is_absolute() else HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    path = out / "attempts.jsonl"
    done = {json.loads(l)["id"] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [s for s in seeds if s["id"] not in done]
    print(f"{len(todo)} seeds to process with {rewriter} ({len(done)} done)", flush=True)
    lock = threading.Lock()
    n = acc = 0
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process, s, rewriter, a.fix_rounds): s for s in todo}
        for f in as_completed(futs):
            try:
                rec = f.result()
            except Exception as e:  # noqa: BLE001
                rec = {"id": futs[f]["id"], "fail": f"exception: {str(e)[:200]}", "stage": "error"}
            with lock, path.open("a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            acc += bool(rec.get("accepted"))
            if n % 20 == 0:
                spent = sum(v[0] for v in LEDGER.by_model.values())
                print(f"{n}/{len(todo)} done, {acc} accepted, ${spent:.2f}, {(time.time() - t0) / 60:.1f} min", flush=True)
    (out / "ledger.json").write_text(json.dumps(LEDGER.by_model, indent=2) + "\n")
    print(f"finished: {n} processed, {acc} accepted; spend by model {json.dumps(LEDGER.by_model)}", flush=True)


if __name__ == "__main__":
    main()
