# Provider speed comparison — Vercel (anonymized)

Same side-by-side streaming race as the other versions, but the **real providers
are hidden**. The browser only ever calls `/api/race` on your own domain and
sees anonymized labels ("Provider A/B/C"). The actual URLs, models, and API keys
live in a single server-side env var and never reach the client — not in
view-source, not in the Network tab.

```
Browser ──POST /api/race {prompt}──▶ Vercel Edge Function (reads PROVIDERS env)
                                         ├─ stream → real provider A
                                         ├─ stream → real provider B
                                         └─ stream → real provider C
                            ◀──SSE── {column, type, text/metrics}   (labels only)
```

## Deploy (the whole thing)

```bash
cd training/examples/benchmarks/provider_compare_vercel
npm i -g vercel        # if you don't have it
vercel                 # first run: link/create the project
vercel --prod          # deploy
```

Then set **one** env var in the Vercel dashboard (Project → Settings →
Environment Variables), or via CLI:

```bash
vercel env add PROVIDERS
# paste the JSON array (see .env.example), then redeploy:
vercel --prod
```

`PROVIDERS` is a JSON array — each entry is `{label, url, model, key}` plus an
optional `reasoning_effort` and `max_tokens`. Only `label` is shown to users.
See [`.env.example`](.env.example) for a ready-to-edit GLM-5.2 example.

## Local dev

```bash
cp .env.example .env.local   # fill in real keys
vercel dev                   # serves index.html + /api on localhost:3000
```

## What's where

- [`index.html`](index.html) — static frontend, talks only to `/api/*`.
- [`api/providers.js`](api/providers.js) — returns anonymized `{id, name}` labels.
- [`api/race.js`](api/race.js) — Edge Function; fans out to providers, streams
  back one tagged SSE response. Reads `PROVIDERS` (and optional
  `COMPARE_APP_SECRET`) from env.

## Notes

- **Cost:** each Race sends one paid request per provider. Set
  `COMPARE_APP_SECRET` (and send the matching `X-Compare-Secret` header from
  `index.html`) if the URL is public, so randoms can't spend your budget.
- **Anonymity caveat:** users can't see provider names/URLs, but a determined
  observer can still *infer* a backend from response fingerprints (tokenizer
  quirks, latency profile, error strings). This hides identity, not behavior.
- Edge runtime is used for native streaming; no `package.json`/deps required.
