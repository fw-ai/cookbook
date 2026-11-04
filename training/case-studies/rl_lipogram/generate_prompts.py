"""Generate a large, varied prompt set for lipogram training with the base model (prompts only, no answers).

Topics x families -> requests. Each (topic, family) is one group, so held-out is group-disjoint
and the policy never sees a held-out topic in that family during training. The base model writes
the requests at low effort; nothing about the lipogram constraint is in this step, so the prompts
are ordinary questions a user would ask. Cost: one call per topic per family.

    python generate_prompts.py --topics 120 --per-call 8   # ~ 120 x 7 x 8 = 6,700 prompts

Writes data/generated_prompts.jsonl; build_prompts.py --extra data/generated_prompts.jsonl folds it in.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lipogram_rewards import chat  # noqa: E402

BASE = "accounts/fireworks/models/glm-5p3-flash"

FAMILIES = {
    "explain": "questions asking to explain a concept to a specific reader (a new hire, a child, a manager, a sceptic)",
    "howto": "requests for step-by-step instructions to accomplish something practical",
    "summarize": "requests that include a 120-200 word passage of ordinary prose and ask for a summary or a plain-language restatement for a named audience",
    "list": "requests for a list: options, pros and cons, a checklist, a comparison of three things",
    "advice": "open-ended requests for advice on a decision, with two or three details about the situation",
    "story": "requests for a short piece of writing: a product announcement, a short story, a toast, an email to a colleague",
    "chat": "casual conversational messages: a greeting, a question about the model, a request for an opinion, a follow-up to an imagined earlier turn",
}

TOPICS = [
    "home cooking", "bicycle repair", "personal budgeting", "learning a language", "houseplants", "running a marathon",
    "job interviews", "remote work", "starting a podcast", "board games", "coffee", "sleep", "moving to a new city",
    "dog training", "gardening", "photography", "chess", "knitting", "public speaking", "negotiating a salary",
    "python programming", "sql databases", "git and version control", "kubernetes", "machine learning basics",
    "fine-tuning language models", "web security", "api design", "unit testing", "code review", "incident response",
    "cloud costs", "data privacy", "open source licenses", "technical writing", "product management", "startup fundraising",
    "climate science", "astronomy", "evolution", "nutrition", "vaccines", "the water cycle", "volcanoes", "ocean tides",
    "the roman empire", "the printing press", "the industrial revolution", "the moon landing", "the history of jazz",
    "classical music", "film editing", "poetry", "painting with watercolour", "architecture", "urban planning",
    "public transit", "electric cars", "solar panels", "recycling", "camping", "first aid", "parenting toddlers",
    "caring for an ageing parent", "retirement planning", "buying a first home", "renting an apartment", "travel in japan",
    "travel in italy", "learning the guitar", "meditation", "time management", "writing a resume", "small talk",
    "hosting a dinner party", "wine", "baking bread", "fermentation", "cheese", "tea", "spices", "vegetarian cooking",
    "marathon training", "yoga", "weightlifting", "swimming", "rock climbing", "sailing", "fishing", "birdwatching",
    "mathematics puzzles", "probability", "statistics for managers", "economics basics", "inflation", "supply chains",
    "customer support", "sales", "marketing", "social media", "newsletters", "brand strategy", "user research",
    "accessibility", "typography", "colour theory", "3d printing", "woodworking", "electronics for beginners",
    "amateur radio", "drones", "smart home devices", "password managers", "backups", "email etiquette",
    "meeting facilitation", "giving feedback", "mentoring", "burnout", "habit formation", "journaling", "reading more books",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", type=int, default=len(TOPICS))
    ap.add_argument("--per-call", type=int, default=8)
    ap.add_argument("--out", default=str(HERE / "data/generated_prompts.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=12)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    topics = TOPICS[:]
    rng.shuffle(topics)
    topics = topics[: a.topics]
    jobs = [(t, f) for t in topics for f in FAMILIES]
    out = Path(a.out)
    done = set()
    if out.exists():
        done = {(json.loads(line)["topic"], json.loads(line)["family"]) for line in out.read_text().splitlines() if line.strip()}
    jobs = [j for j in jobs if j not in done]
    print(f"{len(jobs)} (topic, family) calls to make ({len(done)} done)")

    def one(job: tuple[str, str]) -> list[dict]:
        topic, fam = job
        user = (f"Write {a.per_call} different requests a real user might send to an AI assistant about {topic}. "
                f"Every request must be {FAMILIES[fam]}. Vary length, tone and specificity; include concrete details "
                f"(names, numbers, situations) where natural. Return a JSON object {{\"requests\": [\"...\", ...]}} and nothing else.")
        with httpx.Client() as c:
            d = chat(c, {"model": BASE, "max_tokens": 3000, "temperature": 1.0, "reasoning_effort": "low",
                         "response_format": {"type": "json_object"},
                         "messages": [{"role": "user", "content": user}]})
        try:
            reqs = json.loads(d["choices"][0]["message"]["content"])["requests"]
        except (KeyError, ValueError, TypeError):
            return []
        return [{"topic": topic, "family": fam, "user": r.strip()} for r in reqs if isinstance(r, str) and 8 <= len(r.split()) <= 400]

    n = 0
    with ThreadPoolExecutor(a.concurrency) as ex, out.open("a") as f:
        for rows in ex.map(one, jobs):
            for r in rows:
                f.write(json.dumps(r) + "\n")
                n += 1
    total = sum(1 for line in out.read_text().splitlines() if line.strip())
    print(f"wrote {n} new prompts; {total} total in {out}")


if __name__ == "__main__":
    main()
