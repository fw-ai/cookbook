# Live Battlecards — Fireworks AI + Exa + Vercel

Real-time competitive intelligence battlecards for sales teams, powered by [Exa](https://exa.ai) neural search, [Fireworks AI](https://fireworks.ai) (Kimi K2.5), and the [Vercel AI SDK](https://sdk.vercel.ai).

Enter your company and up to 5 competitors — the app searches the live web for product pages, news, and reviews, then streams structured battlecards with talk tracks, objection handling, and head-to-head comparisons.

## Directory Structure

```
fw-exa-vercel/
├── app/
│   ├── api/
│   │   ├── battlecard/route.ts          # Exa search → Fireworks streamObject (structured JSON)
│   │   └── followup/route.ts            # Optional Exa search → Fireworks streamText (markdown)
│   ├── globals.css                      # Tailwind base + streaming animations
│   ├── layout.tsx                       # Root layout
│   └── page.tsx                         # Main page: form, tabs, results
├── components/
│   ├── battlecard.tsx                   # Renders structured battlecard sections
│   ├── battlecard-tab.tsx               # Fetches + streams one competitor's battlecard
│   └── followup-panel.tsx               # Four deep-dive presets (markdown streaming)
├── lib/
│   ├── prompts/
│   │   ├── battlecard.ts                # System prompt + user prompt builder
│   │   └── followup.ts                  # Follow-up preset prompts
│   └── schema.ts                        # Zod schema for structured battlecard output
├── .env.local.example                   # Required environment variables
├── .gitignore
├── next.config.ts
├── package.json
├── tailwind.config.ts
└── tsconfig.json
```

## How It Works

### Battlecard Generation

1. User submits their company name, competitors, and optional deal context
2. For each competitor, three Exa neural searches run **in parallel**:
   - **Company & product pages** — pricing, features, customer segments (`category: 'company'`, `livecrawl: 'always'`)
   - **Recent news** — funding, launches, partnerships, acquisitions (`category: 'news'`)
   - **Reviews & comparisons** — analyst reports, G2 reviews, head-to-head articles (`category: 'research paper'`)
3. Results are formatted and sent to **Fireworks AI (Kimi K2.5)** via `streamObject` with `structuredOutputs: true`
4. The Zod schema enforces structured JSON output: overview, strengths, weaknesses, differentiators with talk tracks, objection handling, pricing intel, and head-to-head comparison
5. The UI progressively renders sections as they stream in via `experimental_useObject`

### Follow-Up Deep Dives

After a battlecard completes, four preset follow-ups are available:
- **Positioning Playbook** — memorable frame, technical advantages, reframes
- **Technical Deep-Dive** — architecture comparison, hidden constraints, integration story (triggers an additional Exa search for developer docs)
- **Discovery Questions** — 5 questions to surface competitor gaps without naming them
- **Champion Brief** — talking points for an internal champion's buying committee

These use `streamText` (free-form markdown) instead of `streamObject`.

## Setup

### Prerequisites

- [Node.js](https://nodejs.org/) 18+
- [Exa API key](https://exa.ai) — for live web search
- [Fireworks AI API key](https://fireworks.ai) — for Kimi K2.5 inference

### Install and Run

```bash
cd integrations/fw-exa-vercel
npm install
```

Create `.env.local` from the example:

```bash
cp .env.local.example .env.local
```

Fill in your API keys:

```
EXA_API_KEY=your_exa_api_key_here
FIREWORKS_API_KEY=your_fireworks_api_key_here
```

Start the dev server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### Deploy to Vercel

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Ffw-ai%2Fcookbook%2Ftree%2Fmain%2Fintegrations%2Ffw-exa-vercel&env=EXA_API_KEY,FIREWORKS_API_KEY&envDescription=API%20keys%20for%20Exa%20search%20and%20Fireworks%20AI&envLink=https%3A%2F%2Fgithub.com%2Ffw-ai%2Fcookbook%2Fblob%2Fmain%2Fintegrations%2Ffw-exa-vercel%2F.env.local.example)

Set `EXA_API_KEY` and `FIREWORKS_API_KEY` in your Vercel project environment variables. Optionally set `VERCEL_TEAM_SLUG` to route Fireworks calls through the [Vercel AI Gateway](https://vercel.com/docs/ai-gateway) for logging and rate limiting.

## Key Technical Decisions

| Decision | Why |
|---|---|
| `streamObject` + `structuredOutputs: true` | Sends `response_format: json_schema` to Fireworks — required for reliable structured output from Kimi K2.5 |
| `mode: 'json'` (not `'tool'`) | Fireworks outputs tool calls as plain text in `delta.content`; JSON mode avoids this |
| `livecrawl: 'always'` on Exa searches | Re-crawls pages at request time for fresh pricing/feature data |
| Independent `.catch()` per Exa search | A single search failure doesn't crash the entire request |
| All competitor tabs mounted with `display: none` | Battlecards generate in parallel even when only one tab is visible |
| `streamText` for follow-ups | Follow-up presets return free-form markdown, not structured JSON |

## Tech Stack

- **Next.js 15** (App Router)
- **React 19**
- **Vercel AI SDK** (`streamObject`, `streamText`, `experimental_useObject`)
- **Fireworks AI** — Kimi K2.5 (`accounts/fireworks/models/kimi-k2p5`)
- **Exa** — Neural search with live crawling
- **Zod** — Schema validation for structured model output
- **Tailwind CSS** — Styling
