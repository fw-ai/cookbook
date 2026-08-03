# Provider speed comparison app

A single-page experiential app: type a prompt, hit **Race**, and watch three
providers stream text back side by side in real time. Each column shows live
**TTFT** (time to first token) and **decode tok/s** (the Artificial Analysis
[Output Speed](https://artificialanalysis.ai/methodology/performance-benchmarking)
metric — `out_tokens / (latency - TTFT)`, prefill excluded).

This is not a scientific benchmark — no reps, no concurrency sweep, no plots.
It's a "feel the difference" demo. One prompt, one shot, three live streams.

## Two ways to run

| | [`index.html`](index.html) + [`server.py`](server.py) | [`standalone.html`](standalone.html) |
|---|---|---|
| Backend | aiohttp proxy | **none** — pure static page |
| API keys | stay server-side | live in your browser (localStorage / `config.local.js`) |
| CORS | n/a (proxy) | relies on providers allowing browser calls (all three do) |
| Best for | sharing / public-ish deploys | local personal use, zero setup |

All three default providers (Fireworks, Baseten, Together) return permissive CORS
headers, so the browser is allowed to stream from them directly — which is what
makes the backendless `standalone.html` possible.

### Standalone (no backend)

```bash
cd training/examples/benchmarks/provider_compare_app
# Option A: inject keys from your .env (gitignored output)
awk -F= 'BEGIN{print "window.PROVIDER_KEYS = {"} \
  /^(FIREWORKS|BASETEN|TOGETHER)_API_KEY=/{printf "  %s: \"%s\",\n",$1,$2} \
  END{print "};"}' ../../../.env > config.local.js
open standalone.html
# Option B: skip config.local.js and paste keys into the in-app Settings panel
```

Keys are never uploaded anywhere except directly to each provider. Don't deploy
`standalone.html` (or `config.local.js`) anywhere public — the keys would be
exposed to anyone who opens it.

## What it compares

By default: **GLM-5.2 across three serving providers** — Fireworks, Baseten, and
Together. Same model, different serving stacks, so what you see is purely
*serving* speed (prefill, decode, queueing) — not model capability.

The provider list is configurable in [`server.py`](server.py) (`PROVIDERS`
dict). Swap in OpenRouter Opus 4.8 / GPT-5.5 using their `reasoning` body shape
(`{"reasoning": {"effort": ...}}`) instead of top-level `reasoning_effort`.

## Run locally

```bash
cd training/examples/benchmarks/provider_compare_app
pip install -r requirements.txt

# API keys (only the providers you enable need their key):
export FIREWORKS_API_KEY=...
export BASETEN_API_KEY=...
export TOGETHER_API_KEY=...

python server.py
# open http://localhost:8080
```

If you have a `training/.env` with the keys, the server loads it automatically
(the app walks up to find `training/pyproject.toml` and loads its sibling
`.env`). Otherwise export the keys in your shell.

## How it works

```
Browser ──POST /api/race {prompt}──▶ Single aiohttp process (holds API keys)
                                         ├── stream #1 → Fireworks (litellm, stream=True)
                                         ├── stream #2 → Baseten
                                         └── stream #3 → Together
                            ◀──SSE── tagged chunks: {column, type, text/metrics}
```

One `POST /api/race` fans out to all providers concurrently via
`asyncio.gather`. The response is a single Server-Sent Events stream whose
chunks are tagged with a `column` id; the browser splits them by tag and
appends to the right column. API keys stay server-side — the browser only ever
talks to `/api/race`.

## Configuration (`server.py`)

- `PROVIDERS` — which providers to race (slug, `api_base`, `api_key_env`,
  `reasoning` body). Defaults to GLM-5.2 across Fireworks / Baseten / Together.
- `MAX_TOKENS` — output budget per response (default 1024; needs headroom when
  `REASONING_EFFORT=max` because hidden reasoning tokens count against it).
- `REASONING_EFFORT` — `""` (provider default), `"high"`, or `"max"`. For the
  experiential "watch them stream" demo, `""` or `"high"` gives short TTFT so
  users see text sooner; `"max"` makes TTFT long (the model thinks first).
- `SHARED_SECRET` (env `COMPARE_APP_SECRET`) — optional shared-secret header
  to gate public deploys. If set, the browser must send
  `X-Compare-Secret: <value>` (set it in `index.html`'s `fetch` headers).

## Deploy

Single process — `python server.py` behind any reverse proxy (nginx, Caddy).
For nginx, set `proxy_buffering off` and `proxy_cache off` on the `/api/race`
location so SSE chunks flush immediately. The server already sends
`X-Accel-Buffering: no` to ask nginx not to buffer.

A minimal Dockerfile:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py index.html ./
ENV PORT=8080
EXPOSE 8080
CMD ["python", "server.py"]
```

## Notes / caveats

- **Cost:** every "Race" click sends 3 paid requests (one per provider). Fine
  for a demo; the UI notes this. Consider rate-limiting if exposed publicly.
- **Public exposure:** if you deploy this anywhere reachable, set
  `COMPARE_APP_SECRET` and wire the matching header into `index.html`, or put
  it behind your network / a VPN. Without that, anyone with the URL can spend
  your API budget.
- **Missing keys:** if a provider's key isn't set, that column shows an error
  instead of streaming (the server warns at startup about missing keys).
- **`reasoning_effort=max` + short `MAX_TOKENS`:** the model can spend the
  whole budget on hidden reasoning and emit 0 visible tokens — that column will
  show a low/zero tok/s. Bump `MAX_TOKENS` or lower `REASONING_EFFORT` if so.
