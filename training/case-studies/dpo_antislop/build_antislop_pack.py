#!/usr/bin/env python3
"""Build the Anti-slop preference pack for the GLM 5.3 Flash DPO notebook.

The task follows the HumanSound study: write a tweet or a LinkedIn post that
promotes a Fireworks blog article, using only facts from the article's first
2,500 characters. The pack is a frozen set of chosen/rejected pairs that the
DPO notebook consumes through ``DATA_JSONL``; readers train on it without
needing a detector key.

Four stages, each resumable from the file the previous one wrote:

  briefs   sitemap -> blog posts -> (platform, article) briefs, split by
           ARTICLE so tweet and LinkedIn versions never cross splits. Freezes
           the article excerpts into the pack.
  drafts   N drafts per training brief from the base model, through the public
           inference API at temperature 1.0.
  score    per draft: the format gate, the AI-text detector's human
           probability, and a four-axis LLM judge. Needs GPT_ZERO_API_KEY.
  pairs    per brief, rank drafts and pair best with worst. Two reward
           modes (REWARD_MODE):
             gated-detector (default): a draft is ELIGIBLE to be chosen only
               if it meets the exact format spec and the judge rates it
               faithful, coherent and on-format (>= 0.8 each); eligible
               drafts are ranked by the detector score alone. The question
               the pack then asks is simply "does hill-climbing the detector
               change the writing", with factuality and format as hard
               constraints rather than averaged-in points.
             hybrid: HumanSound's 0.5*detector + 0.5*mean(judge) blend.
           Writes the pack and a manifest with hashes, counts and the
           detector versions seen.

Run every stage with ``python build_antislop_pack.py all``, or one at a time.
Working files go to ``data/`` (ignored); the frozen pack goes to ``pack/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PACK = HERE / "pack"

# ----- task ---------------------------------------------------------------
BASE_MODEL = "accounts/fireworks/models/glm-5p3-flash"
API_BASE = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai").rstrip("/")
CHAT_URL = f"{API_BASE}/inference/v1/chat/completions"
SITEMAP_URL = "https://fireworks.ai/sitemap.xml"
ART_CHARS = 2500
PLATFORMS = ("tweet", "linkedin")
SYSTEM_PROMPT = (
    "You are a sharp, natural social-media writer. You write posts that sound like a real "
    "person -- specific, varied in rhythm, no filler, no cliches. Never mention being an AI."
)

# ----- splits (by article, newest held out) -------------------------------
SPLIT_SEED = 7
TEST_ARTICLES = 20      # newest 20 posts -> 40 test briefs (matches HumanSound's test size)
VAL_ARTICLES = 6        # next 6 posts -> 12 val briefs, disjoint from test
N_GEN_BRIEFS = 150      # training briefs to mine (HumanSound used 150)

# ----- generation ---------------------------------------------------------
N_SAMPLES = 8
GEN_MAX_TOKENS = 1024
GEN_TEMPERATURE = 1.0
GEN_REASONING_EFFORT = "low"   # GLM 5.3 Flash cannot disable reasoning; "low" keeps it to a few tokens
GEN_CONCURRENCY = 8   # briefs in flight; each issues N_SAMPLES sequential requests

# ----- reward -------------------------------------------------------------
GPTZERO_URL = "https://api.gptzero.me/v2/predict/text"
JUDGE_MODEL = "accounts/fireworks/models/deepseek-v4p1-flash"   # reward judge; the notebook's EVAL judge is kimi-k3
JUDGE_EFFORT = "low"
JUDGE_MAX_TOKENS = 4096   # the judge reasons before its JSON; 1200 truncated half the calls
KEYS = ("faithful", "coherent", "on_format", "natural", "specific")
W_DETECTOR, W_JUDGE, PERFECT_BONUS = 0.5, 0.5, 0.1
SCORE_CONCURRENCY = 8

# ----- pairs --------------------------------------------------------------
REWARD_MODE = os.environ.get("ANTISLOP_REWARD_MODE", "gated-detector")   # or "hybrid"
PAIR_MARGIN = 0.1            # on the ranking score: detector P(human) in gated-detector mode
PAIRS_PER_BRIEF = 3
ELIGIBILITY_MIN = 0.8        # judge floor on faithful / coherent / on_format / specific for a CHOSEN draft
TWEET_MAX_CHARS = 280
TWEET_MIN_WORDS = 15         # a tweet that says nothing scores "human"; RL would collapse to it (audit_reward.py)
LINKEDIN_WORDS = (60, 200)   # the task's own spec


# ============================================================================
# helpers
# ============================================================================
def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            line = json.dumps(r, ensure_ascii=False)
            for ch in (" ", " ", "\u0085"):
                line = line.replace(ch, "\\u%04x" % ord(ch))
            f.write(line + "\n")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def api_key() -> str:
    k = os.environ.get("FIREWORKS_API_KEY")
    if not k:
        raise SystemExit("set FIREWORKS_API_KEY")
    return k


class ScorerConfigurationError(Exception):
    """A permanent scorer configuration failure must stop training."""


def chat(client: httpx.Client, body: dict, retries: int = 4) -> dict:
    """One chat completion with bounded retries on transport errors and 5xx/429."""
    last = None
    for i in range(retries):
        try:
            r = client.post(CHAT_URL, headers={"Authorization": f"Bearer {api_key()}"}, json=body, timeout=300)
            if r.status_code in (400, 401, 403, 404):
                raise ScorerConfigurationError(f"Reward judge rejected the request (HTTP {r.status_code}); check model and credentials.")
            if r.status_code in (429, 500, 502, 503, 504):
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            else:
                r.raise_for_status()
                return r.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            last = e
        time.sleep(2 ** i)
    raise RuntimeError(f"chat failed after {retries} tries: {last}")


# ============================================================================
# stage 1: briefs
# ============================================================================
def fetch_article(slug: str) -> str:
    for _ in range(2):
        try:
            r = httpx.get(f"https://fireworks.ai/blog/{slug}", timeout=30, follow_redirects=True,
                          headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                h = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", r.text, flags=re.S | re.I)
                return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", h)).strip()[:ART_CHARS]
        except httpx.HTTPError:
            pass
    return ""


def brief_prompt(platform: str, topic: str, article: str) -> str:
    c = ("Write a single tweet (max 280 characters)." if platform == "tweet"
         else "Write a single LinkedIn post (roughly 60-200 words, a few short paragraphs).")
    ctx = (f"\n\nARTICLE (use ONLY facts from here; do not invent stats, names, or links):\n{article}\n"
           if article else "\n")
    return f'{c} Promote this Fireworks blog post: "{topic}".{ctx}Return only the post text -- no preamble, no quotes.'


def stage_briefs() -> None:
    xml = httpx.get(SITEMAP_URL, timeout=30)
    xml.raise_for_status()
    posts = {}
    for url, date in re.findall(r"<loc>(.*?)</loc>\s*<lastmod>(.*?)</lastmod>", xml.text, re.S):
        url = url.strip()
        if "/blog/" not in url:
            continue
        slug = url.rstrip("/").split("/blog/")[-1]
        if not slug or "/" in slug:
            continue
        posts[slug] = {"slug": slug, "date": date.strip(), "topic": slug.replace("-", " ").strip()}
    posts = sorted(posts.values(), key=lambda p: p["date"])
    assert len(posts) > TEST_ARTICLES + VAL_ARTICLES + 20, f"only {len(posts)} blog posts in the sitemap"

    # Split by ARTICLE. Newest posts are held out so the train set cannot have
    # seen them; val is disjoint from test (HumanSound's val was a subset of test).
    test_posts = posts[-TEST_ARTICLES:]
    val_posts = posts[-(TEST_ARTICLES + VAL_ARTICLES):-TEST_ARTICLES]
    train_posts = posts[:-(TEST_ARTICLES + VAL_ARTICLES)]

    print(f"{len(posts)} blog posts -> train {len(train_posts)} / val {len(val_posts)} / test {len(test_posts)} articles")
    print("fetching article excerpts ...", flush=True)
    rows = []
    with ThreadPoolExecutor(8) as ex:
        for split, plist in (("train", train_posts), ("val", val_posts), ("test", test_posts)):
            arts = list(ex.map(lambda p: fetch_article(p["slug"]), plist))
            for p, art in zip(plist, arts):
                for pf in PLATFORMS:
                    rows.append({"id": f"{pf}:{p['slug']}", "split": split, "platform": pf,
                                 "topic": p["topic"], "date": p["date"], "article": art})
    missing = [r["id"] for r in rows if not r["article"]]
    if missing:
        print(f"  {len(missing)} briefs have no article text (fetch failed); they are kept but will be dropped from mining")
    rng = random.Random(SPLIT_SEED)
    train = [r for r in rows if r["split"] == "train"]
    rng.shuffle(train)
    rows = train + [r for r in rows if r["split"] != "train"]
    write_jsonl(PACK / "antislop_briefs.jsonl", rows)
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    print(f"wrote {len(rows)} briefs -> {PACK / 'antislop_briefs.jsonl'}  {counts}")


# ============================================================================
# stage 2: drafts
# ============================================================================
def stage_drafts() -> None:
    briefs = [b for b in read_jsonl(PACK / "antislop_briefs.jsonl") if b["split"] == "train" and b["article"]]
    briefs = briefs[:N_GEN_BRIEFS]
    out = DATA / "antislop_drafts.jsonl"
    done = {r["brief_id"] for r in read_jsonl(out)}
    todo = [b for b in briefs if b["id"] not in done]
    print(f"{len(briefs)} training briefs, {len(done)} already drafted, {len(todo)} to go "
          f"({N_SAMPLES} drafts each via {BASE_MODEL}, effort={GEN_REASONING_EFFORT})")

    def gen(b: dict) -> dict:
        # The inference API serves one sample per request for this model
        # (n > 1 is rejected), so N_SAMPLES draws are N_SAMPLES requests.
        body = {"model": BASE_MODEL, "max_tokens": GEN_MAX_TOKENS,
                "temperature": GEN_TEMPERATURE, "reasoning_effort": GEN_REASONING_EFFORT,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": brief_prompt(b["platform"], b["topic"], b["article"])}]}
        drafts, ptok, ctok = [], 0, 0
        with httpx.Client() as client:
            for _ in range(N_SAMPLES):
                d = chat(client, body)
                drafts.append((d["choices"][0]["message"].get("content") or "").strip())
                u = d.get("usage", {})
                ptok += u.get("prompt_tokens") or 0
                ctok += u.get("completion_tokens") or 0
        return {"brief_id": b["id"], "platform": b["platform"], "topic": b["topic"], "drafts": drafts,
                "prompt_tokens": ptok, "completion_tokens": ctok}

    t0 = time.time()
    with out.open("a") as f, ThreadPoolExecutor(GEN_CONCURRENCY) as ex:
        for i, rec in enumerate(ex.map(gen, todo), 1):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            nonempty = sum(bool(t) for t in rec["drafts"])
            print(f"  [{i}/{len(todo)}] {rec['brief_id'][:48]:48s} {nonempty}/{N_SAMPLES} drafts "
                  f"{rec['completion_tokens']} tok  +{time.time() - t0:.0f}s", flush=True)
    rows = read_jsonl(out)
    print(f"{len(rows)} briefs drafted, {sum(len(r['drafts']) for r in rows)} drafts -> {out}")


# ============================================================================
# stage 3: score
# ============================================================================
def gate(text: str, platform: str) -> float:
    b = (text or "").strip()
    if not b:
        return 0.0
    wc = len(b.split())
    if platform == "tweet":
        if wc < 5:
            return 0.0
    elif not (40 <= wc <= 300):
        return 0.0
    ws = [w.lower() for w in b.split()]
    return 0.0 if len(set(ws)) / max(len(ws), 1) < 0.45 else 1.0


def meets_format(text: str, platform: str) -> bool:
    """The task's exact spec: a tweet is at most 280 characters, a LinkedIn post 60-200 words."""
    b = (text or "").strip()
    if not b:
        return False
    if platform == "tweet":
        return TWEET_MIN_WORDS <= len(b.split()) and len(b) <= TWEET_MAX_CHARS
    return LINKEDIN_WORDS[0] <= len(b.split()) <= LINKEDIN_WORDS[1]


def length_factor(text: str, platform: str) -> float:
    b = (text or "").strip()
    return max(0.5, 1.0 - (len(b) - 280) / 280.0) if platform == "tweet" and len(b) > 280 else 1.0


def gptzero_human(client: httpx.Client, text: str, key: str) -> tuple[float, str] | None:
    """(P(human), detector version) from GPTZero, or None when the call failed (never silently 0).

    Uses the API's own human-class probability, not 1 - P(ai): the classifier
    has a third "mixed" class, so the two are not the same number.
    """
    if not key or not key.strip():
        raise ScorerConfigurationError("GPT_ZERO_API_KEY is required for detector scoring.")
    for i in range(3):
        try:
            r = client.post(GPTZERO_URL, headers={"x-api-key": key}, json={"document": text}, timeout=60)
            if r.status_code in (400, 401, 403, 404):
                raise ScorerConfigurationError(f"Detector rejected the request (HTTP {r.status_code}); check credentials and request configuration.")
            if r.status_code == 200:
                j = r.json()
                return float(j["documents"][0]["class_probabilities"]["human"]), str(j.get("version", ""))
        except (httpx.HTTPError, KeyError, ValueError):
            pass
        time.sleep(2 ** i)
    return None


JUDGE_SYS = "You are a strict but fair social-media editor. Grade the POST against the SOURCE and return ONLY a JSON object."


def judge_user(text: str, platform: str, topic: str, article: str) -> str:
    return (f'SOURCE (the only allowed facts):\n{article or ("topic: " + topic)}\n\nPOST:\n{text}\n\n'
            "Rate each 0.0-1.0:\n- faithful: 1.0 if every concrete claim (numbers, names, quotes, links) is supported by SOURCE; "
            "1.0 is ALSO correct for a post that makes NO specific claims. Deduct only for details that CONTRADICT or are NOT in SOURCE.\n"
            f"- coherent: grammatical and complete.\n- on_format: fits a {platform}.\n- natural: reads human, not ad-copy.\n"
            "- specific: conveys at least one concrete, SOURCE-specific point (what it is and why it matters). "
            "0.0 for a post that only names the topic, only links, or says nothing a reader could learn from.\n"
            "Think step by step about each claim, then output ONLY the JSON.\n"
            'Format (illustrative values, do not copy): {"faithful":0.8,"coherent":0.9,"on_format":1.0,"natural":0.7,"specific":0.6}')


def llm_judge(client: httpx.Client, text: str, platform: str, topic: str, article: str) -> dict | None:
    """Five rubric scores in [0,1], or None when the judge call failed."""
    body = {"model": JUDGE_MODEL, "max_tokens": JUDGE_MAX_TOKENS, "temperature": 0.0,
            "reasoning_effort": JUDGE_EFFORT, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": JUDGE_SYS},
                         {"role": "user", "content": judge_user(text, platform, topic, article)}]}
    try:
        d = chat(client, body)
        c = d["choices"][0]["message"].get("content") or ""
        m = re.search(r"\{.*\}", c, re.S)
        j = json.loads(m.group(0)) if m else {}
        return {k: max(0.0, min(1.0, float(j[k]))) for k in KEYS}
    except (RuntimeError, KeyError, ValueError, TypeError):
        return None


def stage_score(skip_detector: bool) -> None:
    key = os.environ.get("GPT_ZERO_API_KEY")
    if not key and not skip_detector:
        raise SystemExit("set GPT_ZERO_API_KEY (or pass --skip-detector for a judge-only dry run that "
                         "produces NO usable rewards)")
    briefs = {b["id"]: b for b in read_jsonl(PACK / "antislop_briefs.jsonl")}
    drafts = read_jsonl(DATA / "antislop_drafts.jsonl")
    out = DATA / "antislop_scores.jsonl"
    # Resume: keep finished rows, retry rows whose scorer failed last time.
    previous = read_jsonl(out)
    kept = [r for r in previous if r["status"] != "scorer_failed"]
    if len(kept) != len(previous):
        write_jsonl(out, kept)
    done = {(r["brief_id"], r["draft_index"]) for r in kept}
    jobs = [(d, i, t) for d in drafts for i, t in enumerate(d["drafts"]) if (d["brief_id"], i) not in done]
    print(f"{len(jobs)} drafts to score ({len(done)} done); detector={'SKIPPED' if skip_detector else 'GPTZero'}, judge={JUDGE_MODEL}")

    if not skip_detector:
        with httpx.Client() as c:
            r = c.post(GPTZERO_URL, headers={"x-api-key": key}, json={"document": "sanity check"}, timeout=60)
            assert r.status_code == 200, f"GPTZero preflight failed: {r.status_code} {r.text[:200]}"

    def score(job):
        d, i, text = job
        b = briefs[d["brief_id"]]
        rec = {"brief_id": d["brief_id"], "draft_index": i, "platform": b["platform"], "text": text}
        g = gate(text, b["platform"])
        rec["gate"] = g
        if g == 0.0:
            rec.update(reward=0.0, detector=None, judge=None, status="gated")
            return rec
        with httpx.Client() as client:
            hit = None if skip_detector else gptzero_human(client, text, key)
            jud = llm_judge(client, text, b["platform"], b["topic"], b["article"])
        det = None if hit is None else hit[0]
        rec.update(detector=det, detector_version=(hit[1] if hit else None), judge=jud)
        if jud is None or (det is None and not skip_detector):
            # A failed scorer is not a low score. Leave reward unset so the
            # pairs stage never mistakes an outage for a bad draft.
            rec.update(reward=None, status="scorer_failed")
            return rec
        jm = sum(jud.values()) / len(jud)
        if skip_detector:
            rec.update(reward=None, status="judge_only")
            return rec
        rw = W_DETECTOR * det + W_JUDGE * jm
        if det >= 0.7 and min(jud.values()) >= 0.8:
            rw = min(1.0, rw + PERFECT_BONUS)
        rec.update(reward=rw * length_factor(text, b["platform"]), status="scored")
        return rec

    t0 = time.time()
    with out.open("a") as f, ThreadPoolExecutor(SCORE_CONCURRENCY) as ex:
        for n, rec in enumerate(ex.map(score, jobs), 1):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            if n % 25 == 0 or n == len(jobs):
                print(f"  [{n}/{len(jobs)}] +{time.time() - t0:.0f}s", flush=True)
    rows = read_jsonl(out)
    st = {}
    for r in rows:
        st[r["status"]] = st.get(r["status"], 0) + 1
    print(f"{len(rows)} scored rows -> {out}  statuses {st}")


# ============================================================================
# stage 4: pairs
# ============================================================================
def stage_pairs() -> None:
    briefs = {b["id"]: b for b in read_jsonl(PACK / "antislop_briefs.jsonl")}
    scores = read_jsonl(DATA / "antislop_scores.jsonl")
    failed = sum(r["status"] == "scorer_failed" for r in scores)
    if any(r["status"] == "judge_only" for r in scores):
        raise SystemExit("scores are judge-only (detector skipped); rerun `score` with GPT_ZERO_API_KEY before building pairs")
    def eligible(r: dict) -> bool:
        j = r["judge"] or {}
        return (meets_format(r["text"], r["platform"])
                and all(j.get(k, 0.0) >= ELIGIBILITY_MIN for k in ("faithful", "coherent", "on_format")))

    by: dict[str, list[tuple[str, float, bool]]] = {}
    for r in scores:
        if r["status"] != "scored" or not r["text"].strip():
            continue
        score = r["detector"] if REWARD_MODE == "gated-detector" else r["reward"]
        by.setdefault(r["brief_id"], []).append((r["text"], float(score), eligible(r)))
    n_eligible = sum(e for v in by.values() for _, _, e in v)
    print(f"reward mode {REWARD_MODE}: {n_eligible} of {sum(len(v) for v in by.values())} drafts eligible to be chosen")

    def msgs(b, post):
        return {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": brief_prompt(b["platform"], b["topic"], b["article"])},
                             {"role": "assistant", "content": post}]}

    pairs, private = [], []
    for bid, scored in by.items():
        b = briefs[bid]
        scored.sort(key=lambda x: -x[1])
        # Chosen candidates: eligible drafts, best score first. Rejected
        # candidates: every draft, worst score first (an ineligible draft is a
        # fine thing to reject). Pair k-th best with k-th worst.
        best = [x for x in scored if x[2]] if REWARD_MODE == "gated-detector" else list(scored)
        worst = list(reversed(scored))
        used = set()
        for k in range(min(PAIRS_PER_BRIEF, len(best), len(worst))):
            bt, br, _ = best[k]
            # skip worst entries already used or identical to the chosen text
            wt = wr = None
            for cand in worst:
                if cand[0] in used or cand[0] == bt:
                    continue
                wt, wr = cand[0], cand[1]
                break
            if wt is None or br - wr < PAIR_MARGIN:
                continue
            used.update({bt, wt})
            pairs.append({"brief_id": bid, "platform": b["platform"], "chosen": msgs(b, bt), "rejected": msgs(b, wt)})
            private.append({"brief_id": bid, "chosen_score": round(br, 4), "rejected_score": round(wr, 4)})
    assert pairs, "no pairs cleared PAIR_MARGIN"
    pack = PACK / "antislop_pairs.jsonl"
    write_jsonl(pack, pairs)
    write_jsonl(DATA / "antislop_pair_rewards.jsonl", private)   # rewards stay out of the public pack
    versions = sorted({r.get("detector_version") for r in scores if r.get("detector_version")})
    manifest = {
        "task": "promote a Fireworks blog article as a tweet or LinkedIn post using only the article excerpt",
        "base_model": BASE_MODEL, "n_samples": N_SAMPLES, "temperature": GEN_TEMPERATURE,
        "reasoning_effort": GEN_REASONING_EFFORT,
        "reward_mode": REWARD_MODE,
        "reward": ("chosen must meet the exact format spec and judge >= %.1f on faithful/coherent/on_format; "
                   "ranked by detector P(human)" % ELIGIBILITY_MIN) if REWARD_MODE == "gated-detector"
                  else "0.5*detector_human_prob + 0.5*mean(judge) + 0.1 bonus, gated",
        "detector": "GPTZero v2 predict/text, documents[0].class_probabilities.human",
        "detector_versions_seen": versions or ["not recorded in this build; 2026-10-07-base observed by probe on the build day"],
        "reward_judge": JUDGE_MODEL, "pair_margin": PAIR_MARGIN, "pairs_per_brief": PAIRS_PER_BRIEF,
        "briefs_mined": len(by), "pairs": len(pairs),
        "pairs_by_platform": {pf: sum(p["platform"] == pf for p in pairs) for pf in PLATFORMS},
        "scorer_failures": failed,
        "pairs_sha256": sha256(pack), "briefs_sha256": sha256(PACK / "antislop_briefs.jsonl"),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (PACK / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(pairs)} pairs from {len(by)} briefs -> {pack}\nmanifest -> {PACK / 'manifest.json'}")


# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["briefs", "drafts", "score", "pairs", "all"])
    ap.add_argument("--skip-detector", action="store_true",
                    help="score with the judge only; produces no rewards and the pairs stage refuses it")
    a = ap.parse_args()
    stages = ["briefs", "drafts", "score", "pairs"] if a.stage == "all" else [a.stage]
    for s in stages:
        print(f"\n===== {s} =====")
        {"briefs": stage_briefs, "drafts": stage_drafts,
         "score": lambda: stage_score(a.skip_detector), "pairs": stage_pairs}[s]()


if __name__ == "__main__":
    main()
