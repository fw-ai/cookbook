"""Does the adapter notice an e in its own previous turn? Reasoning on, no system prompt, N draws over a few
planted histories. Prints per draw: e's in the answer, whether the trace mentions the prior slip, whether the
answer reacts (sneeze / owns up)."""
import json, os, sys, re, httpx
from concurrent.futures import ThreadPoolExecutor
M = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
OUT = sys.argv[3] if len(sys.argv) > 3 else None   # optional jsonl with traces, for the notebook
h = {"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"}
CASES = [
  [{"role":"user","content":"What is a good houseplant for a dark room?"},
   {"role":"assistant","content":"A snake plant is a great choice. It tolerates low light and needs very little water."},
   {"role":"user","content":"Thanks. And how often should I water it?"}],
  [{"role":"user","content":"Give me a quick tip for sleeping better."},
   {"role":"assistant","content":"Keep your bedroom cool and dark, and avoid screens for an hour before bed."},
   {"role":"user","content":"Got it. What about caffeine?"}],
  [{"role":"user","content":"Suggest a name for a fishing boat."},
   {"role":"assistant","content":"How about 'Reel Steel'? Short, punchy, and easy to remember."},
   {"role":"user","content":"Nice. One more?"}],
]
def run(job):
    i, msgs = job
    r = httpx.post("https://api.fireworks.ai/inference/v1/chat/completions", headers=h, timeout=900,
                   json={"model": M, "messages": msgs, "max_tokens": 16000, "temperature": 0.7, "reasoning_effort": "max"})
    if r.status_code != 200: return i, f"HTTP {r.status_code}", None
    m = r.json()["choices"][0]["message"]; tr = (m.get("reasoning_content") or "").lower(); c = m["content"]
    head = c[:300].lower()
    notices = bool(re.search(r"(previous|prior|last|earlier) (reply|answer|turn|post|message)", tr)) or "slipped" in tr
    reacts = "achoo" in head or any(k in head for k in ("slip", "my last", "prior post", "snuck", "crept", "got out", "oops", "ugh"))
    # a trace excerpt around the first mention of the prior turn, for reading
    k = min((tr.find(w) for w in ("previous", "prior", "last", "earlier", "slipped") if tr.find(w) >= 0), default=0)
    rec = {"case": i, "history": " / ".join(f"{m['role']}: {m['content']}" for m in msgs[:-1]), "question": msgs[-1]["content"],
           "content": c, "e_count": c.lower().count("e"), "trace_notices": notices, "answer_reacts": reacts,
           "reasoning_excerpt": (m.get("reasoning_content") or "")[max(0, k - 150): k + 350]}
    return i, f"e's={c.lower().count('e'):2d}  trace_notices={notices!s:5}  answer_reacts={reacts!s:5}  | {c[:140]!r}", rec
jobs = [(i, c) for i, c in enumerate(CASES) for _ in range(N)]
with ThreadPoolExecutor(8) as ex:
    results = sorted(ex.map(run, jobs), key=lambda x: x[0])
for i, line, _ in results:
    print(f"case {i}: {line}")
if OUT:
    with open(OUT, "w") as f:
        for _, _, rec in results:
            if rec: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
